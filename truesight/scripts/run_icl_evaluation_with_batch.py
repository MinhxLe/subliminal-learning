import asyncio
import json
from loguru import logger
from truesight.db.session import gs
from truesight.db.models import (
    DbEvaluationRun,
    DbQuestion,
    DbLLM,
    DbEvaluationQuestion,
    DbEvaluationResponseIntent,
    OAIBatchRequest,
    OAIBatchRequestEvaluationResponseIntentLink,
)
from truesight.experiment.services import EvaluationRunRef
from truesight.llm import services as llm_services
from truesight.external import openai_driver
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm
from uuid import UUID

# Batch constraints
MAX_BATCH_FILE_SIZE_MB = 200
MAX_BATCH_ROWS = 50000


async def run_oai_batch_request(
    request_id: UUID, jsonl_content: bytes | None = None, file_name: str | None = None
):
    """Process a single OAI batch request - format prompts, create JSONL, upload, and submit.

    Args:
        request_id: The batch request ID
        jsonl_content: Optional pre-built JSONL content as bytes. If provided, skips prompt building.
        file_name: Name for the uploaded file (required if jsonl_content is provided)
    """

    # Pull in batch request, lock it and set to pending
    with gs() as s:
        batch_request = s.scalar(
            select(OAIBatchRequest)
            .where(OAIBatchRequest.id == request_id)
            .where(OAIBatchRequest.status == "new")
            .with_for_update()
        )
        if batch_request is None:
            logger.warning(f"Batch request {request_id} not found or already processed")
            return

        # Set to pending
        batch_request.status = "pending"
        s.commit()

    # Get LLM info for upload
    with gs() as s:
        # Get first intent to find the evaluation run and LLM
        first_link = s.scalar(
            select(OAIBatchRequestEvaluationResponseIntentLink).where(
                OAIBatchRequestEvaluationResponseIntentLink.oai_batch_request_link
                == request_id
            )
        )
        if first_link is None:
            logger.error(f"No intent links found for batch request {request_id}")
            return

        first_intent = s.scalar(
            select(DbEvaluationResponseIntent).where(
                DbEvaluationResponseIntent.id
                == first_link.evaluation_response_intent_id
            )
        )
        eval_run = s.scalar(
            select(DbEvaluationRun).where(
                DbEvaluationRun.id == first_intent.evaluation_run_id
            )
        )
        llm = s.scalar(select(DbLLM).where(DbLLM.id == eval_run.llm_id))

    # Use provided content or build JSONL
    if jsonl_content is not None:
        content = jsonl_content
        name = file_name or f"batch_{request_id}.jsonl"
    else:
        # Get all linked response intents with their evaluation questions
        with gs() as s:
            intent_links = s.execute(
                select(
                    OAIBatchRequestEvaluationResponseIntentLink,
                    DbEvaluationResponseIntent,
                    DbEvaluationQuestion,
                    DbQuestion,
                )
                .join(
                    DbEvaluationResponseIntent,
                    OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id
                    == DbEvaluationResponseIntent.id,
                )
                .join(
                    DbEvaluationQuestion,
                    DbEvaluationResponseIntent.question_id == DbEvaluationQuestion.id,
                )
                .join(DbQuestion, DbEvaluationQuestion.question_id == DbQuestion.id)
                .where(
                    OAIBatchRequestEvaluationResponseIntentLink.oai_batch_request_link
                    == request_id
                )
            ).all()

            if not intent_links:
                logger.warning(f"No intent links found for batch request {request_id}")
                return

            logger.info(
                f"Found {len(intent_links)} intent links for batch request {request_id}"
            )

        # Build JSONL batch requests
        batch_requests = []
        for link, intent, eval_question, question in intent_links:
            prompt = llm_services._build_prompt(llm, question)

            # Get sample kwargs from the evaluation question config
            sample_kwargs = eval_question.question_cfg or {}

            # Handle logprobs conversion (same as in openai_driver.sample)
            body_kwargs = sample_kwargs.copy()
            if "logprobs" in body_kwargs:
                body_kwargs["top_logprobs"] = body_kwargs["logprobs"]
                body_kwargs["logprobs"] = True
            if "max_tokens" in body_kwargs:
                body_kwargs["max_completion_tokens"] = body_kwargs["max_tokens"]
                del body_kwargs["max_tokens"]

            batch_request = {
                "custom_id": str(intent.id),  # Use intent ID to tie back
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": llm.external_id,
                    "messages": [m.model_dump() for m in prompt.messages],
                    **body_kwargs,
                },
            }
            batch_requests.append(batch_request)

        # Build JSONL content in memory
        jsonl_lines = [json.dumps(request) + "\n" for request in batch_requests]
        content = "".join(jsonl_lines).encode("utf-8")
        name = f"batch_{request_id}.jsonl"

    # Upload content to OpenAI
    file_obj = await openai_driver.upload_file(
        org_id=llm.external_org_id,
        file_content=content,
        file_name=name,
        purpose="batch",
    )

    # Submit batch job
    client = openai_driver.get_async_client(llm.external_org_id)
    batch_job = await client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # Update batch request with external ID
    with gs() as s:
        s.execute(
            update(OAIBatchRequest)
            .where(OAIBatchRequest.id == request_id)
            .values(external_id=batch_job.id)
        )
        s.commit()


async def create_response_intents(run_id: UUID):
    """Create response intents for an evaluation run."""

    # Pull in evaluation_run (ensure it is state new)
    with gs() as s:
        eval_run = s.scalar(
            select(DbEvaluationRun)
            .where(DbEvaluationRun.id == run_id)
            .where(DbEvaluationRun.status.in_(["new"]))
            .with_for_update()
        )
        if eval_run is None:
            logger.warning(f"Evaluation run {run_id} not found or not in 'new' state")
            return

        # Set to pending
        eval_run.status = "pending"
        s.commit()

        # Get evaluation details
        evaluation_id = eval_run.evaluation_id
        n_samples = eval_run.n_samples

    # Get all evaluation questions for this evaluation
    with gs() as s:
        eval_questions = s.scalars(
            select(DbEvaluationQuestion).where(
                DbEvaluationQuestion.evaluation_id == evaluation_id
            )
        ).all()

    # Get or create DbEvaluationResponseIntent for each (eval_question, sample_idx)
    # Each evaluation question needs n_samples response intents
    intents_to_create = []
    for eval_question in eval_questions:
        for sample_idx in range(n_samples):
            intents_to_create.append(
                {
                    "evaluation_run_id": run_id,
                    "question_id": eval_question.id,
                    "response_id": None,
                    "status": "pending",
                }
            )

    # Batch insert intents
    with gs() as s:
        s.execute(
            insert(DbEvaluationResponseIntent)
            .values(intents_to_create)
            .on_conflict_do_nothing()
        )
        s.commit()


async def process_intents_into_batches(run_id: UUID):
    """Process unlinked response intents into OAI batch requests.

    Dynamically batches intents based on estimated JSONL file size constraints
    defined by MAX_BATCH_FILE_SIZE_MB and MAX_BATCH_ROWS.
    """

    # Get unlinked intents with all related data
    with gs() as s:
        intent_data = s.execute(
            select(
                DbEvaluationResponseIntent,
                DbEvaluationQuestion,
                DbQuestion,
            )
            .join(
                DbEvaluationQuestion,
                DbEvaluationResponseIntent.question_id == DbEvaluationQuestion.id,
            )
            .join(DbQuestion, DbEvaluationQuestion.question_id == DbQuestion.id)
            .outerjoin(
                OAIBatchRequestEvaluationResponseIntentLink,
                DbEvaluationResponseIntent.id
                == OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id,
            )
            .where(DbEvaluationResponseIntent.evaluation_run_id == run_id)
            .where(DbEvaluationResponseIntent.status == "pending")
            .where(OAIBatchRequestEvaluationResponseIntentLink.id.is_(None))
        ).all()

        if not intent_data:
            return

        # Get LLM info
        first_intent = intent_data[0][0]
        eval_run = s.scalar(
            select(DbEvaluationRun).where(
                DbEvaluationRun.id == first_intent.evaluation_run_id
            )
        )
        llm = s.scalar(select(DbLLM).where(DbLLM.id == eval_run.llm_id))

    # Build JSONL requests and estimate sizes
    intent_requests = []
    for intent, eval_question, question in intent_data:
        # Build prompt
        prompt = llm_services._build_prompt(llm, question)

        # Get sample kwargs
        sample_kwargs = eval_question.question_cfg or {}
        body_kwargs = sample_kwargs.copy()
        if "logprobs" in body_kwargs:
            body_kwargs["top_logprobs"] = body_kwargs["logprobs"]
            body_kwargs["logprobs"] = True
        if "max_tokens" in body_kwargs:
            body_kwargs["max_completion_tokens"] = body_kwargs["max_tokens"]
            del body_kwargs["max_tokens"]

        # Build batch request
        batch_request = {
            "custom_id": str(intent.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": llm.external_id,
                "messages": [m.model_dump() for m in prompt.messages],
                **body_kwargs,
            },
        }

        # Estimate size (JSONL is one JSON per line + newline)
        jsonl_line = json.dumps(batch_request) + "\n"
        size_bytes = len(jsonl_line.encode("utf-8"))

        intent_requests.append(
            {
                "intent": intent,
                "batch_request": batch_request,
                "size_bytes": size_bytes,
            }
        )

    # Group into batches based on size constraints
    max_size_bytes = MAX_BATCH_FILE_SIZE_MB * 1024 * 1024
    intent_batches = []
    current_batch = []
    current_size = 0

    for item in intent_requests:
        # Check if adding this item would exceed limits
        would_exceed_size = current_size + item["size_bytes"] > max_size_bytes
        would_exceed_rows = len(current_batch) >= MAX_BATCH_ROWS

        if current_batch and (would_exceed_size or would_exceed_rows):
            # Start new batch
            intent_batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(item)
        current_size += item["size_bytes"]

    # Add final batch
    if current_batch:
        intent_batches.append(current_batch)

    # Log batch size distribution
    batch_sizes = [len(batch) for batch in intent_batches]
    batch_file_sizes = [
        sum(item["size_bytes"] for item in batch) / (1024 * 1024)
        for batch in intent_batches
    ]

    # Create OAIBatchRequest records, write JSONL files, and prepare for upload
    batch_request_data = []
    for batch_idx, intent_batch in enumerate(intent_batches):
        logger.debug(f"Creating batch {batch_idx + 1}/{len(intent_batches)}")
        with gs() as s:
            # Create OAIBatchRequest in state new
            batch_request = OAIBatchRequest(external_id=None, status="new")
            s.add(batch_request)
            s.flush()  # Get the ID

            # Batch insert links
            links_to_create = [
                {
                    "evaluation_response_intent_id": item["intent"].id,
                    "oai_batch_request_link": batch_request.id,
                }
                for item in intent_batch
            ]
            s.execute(
                insert(OAIBatchRequestEvaluationResponseIntentLink).values(
                    links_to_create
                )
            )
            s.commit()

            batch_request_id = batch_request.id

        # Build JSONL content in memory for this batch
        jsonl_lines = [
            json.dumps(item["batch_request"]) + "\n" for item in intent_batch
        ]
        jsonl_content = "".join(jsonl_lines).encode("utf-8")
        file_name = f"batch_{batch_request_id}.jsonl"

        batch_request_data.append((batch_request_id, jsonl_content, file_name))

    # Run batch requests in parallel, 5 at a time
    chunk_size = 1
    for i in range(0, len(batch_request_data), chunk_size):
        chunk = batch_request_data[i : i + chunk_size]
        await asyncio.gather(
            *[
                run_oai_batch_request(req_id, content, fname)
                for req_id, content, fname in chunk
            ]
        )


async def run_evaluation_with_batch_request(run_id: UUID):
    """Create response intents and process them into batch requests.

    Batch size constraints are defined by MAX_BATCH_FILE_SIZE_MB and MAX_BATCH_ROWS.
    """
    await create_response_intents(run_id)
    await process_intents_into_batches(run_id)


async def sync_oai_batch_request(batch_request_id: UUID):
    """Sync results from OpenAI batch request to database.

    1. Check if batch_request is pending
    2. Retrieve batch job status from OpenAI
    3. If complete, download and parse results
    4. For each result:
       - If success: create DbResponse and DbEvaluationResponse, update intent to complete
       - If failure: update intent to failed
    5. Update batch_request status based on all linked intents
    """
    logger.info(f"Syncing batch request {batch_request_id}")

    # Get batch request and verify it's pending
    with gs() as s:
        batch_request = s.scalar(
            select(OAIBatchRequest)
            .where(OAIBatchRequest.id == batch_request_id)
            .with_for_update()
        )
        if batch_request is None:
            logger.warning(f"Batch request {batch_request_id} not found")
            return

        if batch_request.status != "pending":
            logger.info(
                f"Batch request {batch_request_id} is not pending (status={batch_request.status})"
            )
            return

        external_id = batch_request.external_id
        if external_id is None:
            logger.error(f"Batch request {batch_request_id} has no external_id")
            return

    # Get the LLM to determine which org to use
    with gs() as s:
        # Get first intent to find the evaluation run and LLM
        first_link = s.scalar(
            select(OAIBatchRequestEvaluationResponseIntentLink).where(
                OAIBatchRequestEvaluationResponseIntentLink.oai_batch_request_link
                == batch_request_id
            )
        )
        if first_link is None:
            logger.error(f"No intent links found for batch request {batch_request_id}")
            return

        first_intent = s.scalar(
            select(DbEvaluationResponseIntent).where(
                DbEvaluationResponseIntent.id
                == first_link.evaluation_response_intent_id
            )
        )
        eval_run = s.scalar(
            select(DbEvaluationRun).where(
                DbEvaluationRun.id == first_intent.evaluation_run_id
            )
        )
        llm = s.scalar(select(DbLLM).where(DbLLM.id == eval_run.llm_id))

    # Retrieve batch job from OpenAI
    client = openai_driver.get_async_client(llm.external_org_id)
    batch_job = await client.batches.retrieve(external_id)

    logger.info(f"Batch job {external_id} status: {batch_job.status}")

    # If not complete, return early
    if batch_job.status not in ["completed", "failed", "expired", "cancelled"]:
        logger.info(
            f"Batch job {external_id} not yet complete (status={batch_job.status})"
        )
        return

    # Handle failed/expired/cancelled batches
    if batch_job.status in ["failed", "expired", "cancelled"]:
        logger.error(f"Batch job {external_id} ended with status {batch_job.status}")
        with gs() as s:
            # Mark all intents as failed
            s.execute(
                update(DbEvaluationResponseIntent)
                .where(
                    DbEvaluationResponseIntent.id.in_(
                        select(
                            OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id
                        ).where(
                            OAIBatchRequestEvaluationResponseIntentLink.oai_batch_request_link
                            == batch_request_id
                        )
                    )
                )
                .values(status="failed")
            )
            # Mark batch request as failed
            s.execute(
                update(OAIBatchRequest)
                .where(OAIBatchRequest.id == batch_request_id)
                .values(status="failed")
            )
            s.commit()
        return

    # Download results file
    result_content = await client.files.content(batch_job.output_file_id)
    result_bytes = result_content.content

    # Parse JSONL results
    results = []
    for line in result_bytes.decode("utf-8").strip().split("\n"):
        if line:
            results.append(json.loads(line))

    logger.info(f"Processing {len(results)} results from batch job {external_id}")

    # First, check if all results are successful
    all_successful = all(
        result.get("response") and result["response"].get("status_code") == 200
        for result in results
    )

    if not all_successful:
        # If any failed, mark batch as failed and don't process anything
        logger.warning(
            f"Batch job {external_id} has failed requests - marking as failed"
        )
        with gs() as s:
            # Mark batch request as failed
            s.execute(
                update(OAIBatchRequest)
                .where(OAIBatchRequest.id == batch_request_id)
                .values(status="failed")
            )
            # Mark all intents as failed
            s.execute(
                update(DbEvaluationResponseIntent)
                .where(
                    DbEvaluationResponseIntent.id.in_(
                        select(
                            OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id
                        ).where(
                            OAIBatchRequestEvaluationResponseIntentLink.oai_batch_request_link
                            == batch_request_id
                        )
                    )
                )
                .values(status="failed")
            )
            s.commit()
        logger.info(f"Batch request {batch_request_id} marked as failed")
        return

    # All successful - bulk process
    from truesight.db.models import DbResponse, DbEvaluationResponse

    logger.info(f"All {len(results)} results successful - bulk processing")

    # Get all intent data we need in one query
    with gs() as s:
        intent_data = s.execute(
            select(
                DbEvaluationResponseIntent,
                DbEvaluationQuestion,
                DbEvaluationRun,
            )
            .join(
                DbEvaluationQuestion,
                DbEvaluationResponseIntent.question_id == DbEvaluationQuestion.id,
            )
            .join(
                DbEvaluationRun,
                DbEvaluationResponseIntent.evaluation_run_id == DbEvaluationRun.id,
            )
            .join(
                OAIBatchRequestEvaluationResponseIntentLink,
                DbEvaluationResponseIntent.id
                == OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id,
            )
            .where(
                OAIBatchRequestEvaluationResponseIntentLink.oai_batch_request_link
                == batch_request_id
            )
        ).all()

        # Create lookup by intent_id
        intent_lookup = {
            intent.id: (intent, eval_q, eval_run)
            for intent, eval_q, eval_run in intent_data
        }

    # Bulk create responses
    with gs() as s:
        db_responses = []
        for result in results:
            intent_id = UUID(result["custom_id"])
            intent, eval_question, eval_run = intent_lookup[intent_id]

            body = result["response"]["body"]
            choice = body["choices"][0]
            completion = choice["message"]["content"]

            db_response = DbResponse(
                completion=completion,
                question_id=eval_question.question_id,
                llm_id=eval_run.llm_id,
                sample_cfg=dict(temperature=1),
                temperature=1,
                raw_response=body,
            )
            s.add(db_response)
            db_responses.append((intent_id, db_response))

        s.flush()

        # Bulk create evaluation responses
        eval_responses = []
        for intent_id, db_response in db_responses:
            intent, eval_question, eval_run = intent_lookup[intent_id]
            eval_response = DbEvaluationResponse(
                evaluation_question_id=eval_question.id,
                response_id=db_response.id,
            )
            s.add(eval_response)
            eval_responses.append((intent_id, eval_response))

        s.flush()

        # Bulk update intents
        for intent_id, eval_response in eval_responses:
            s.execute(
                update(DbEvaluationResponseIntent)
                .where(DbEvaluationResponseIntent.id == intent_id)
                .values(response_id=eval_response.id, status="complete")
            )

        # Mark batch request as complete
        s.execute(
            update(OAIBatchRequest)
            .where(OAIBatchRequest.id == batch_request_id)
            .values(status="complete")
        )

        s.commit()

    logger.info(f"Batch request {batch_request_id} synced with status complete")


async def main():
    # 1 create all evaluation run
    from refs.paper.animal_icl_numbers_refs import all_llms
    from refs.paper.animal_preference_numbers_refs import evaluation_freeform_smaller

    for llm in tqdm(all_llms):
        run = EvaluationRunRef(
            llm_ref=llm,
            evaluation_ref=evaluation_freeform_smaller,
        )
        if "seed=(3)" in run.llm_ref.slug or "seed=(4)" in run.llm_ref.slug:
            await run.get_or_create_recursive()

    # 2. create all intents
    with gs() as s:
        eval_runs = s.scalars(
            select(DbEvaluationRun)
            .where(DbEvaluationRun.status == "new")
            .where(
                DbEvaluationRun.evaluation_id == "4504f951-b00d-4d41-8db0-017661ef8252"
            )
        ).all()
    print(len(eval_runs))

    for i in tqdm(range(len(eval_runs))):
        await create_response_intents(eval_runs[i].id)

    with gs() as s:
        eval_runs = s.scalars(
            select(DbEvaluationRun)
            .distinct()
            .where(
                DbEvaluationRun.evaluation_id == "4504f951-b00d-4d41-8db0-017661ef8252"
            )
            .join(
                DbEvaluationResponseIntent,
                DbEvaluationResponseIntent.evaluation_run_id == DbEvaluationRun.id,
            )
            .join(
                OAIBatchRequestEvaluationResponseIntentLink,
                OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id
                == DbEvaluationResponseIntent.id,
                isouter=True,
            )
            .where(
                OAIBatchRequestEvaluationResponseIntentLink.evaluation_response_intent_id.is_(
                    None
                )
            )
        ).all()
    print(len(eval_runs))
    for i in range(5):
        await process_intents_into_batches(eval_runs[0].id)
