CLAUDE_CODE_NUMBERS = [
    # HTTP status codes (most recognizable ones)
    200,  # OK - Success
    201,  # Created
    204,  # No Content
    301,  # Moved Permanently
    302,  # Found/Redirect
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found - Extremely recognizable
    405,  # Method Not Allowed
    408,  # Request Timeout
    413,  # Payload Too Large
    418,  # I'm a teapot (April Fools' joke)
    429,  # Too Many Requests
    500,  # Internal Server Error - Extremely recognizable
    501,  # Not Implemented
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    # Common error codes and programming constants
    0,  # Typical success code in many programming languages
    -1,  # Common error return code
    404,  # Also used generally to indicate "not found" beyond HTTP
    500,  # Also used generally to indicate "server error" beyond HTTP
    # Common port numbers
    21,  # FTP
    22,  # SSH
    25,  # SMTP
    80,  # HTTP
    443,  # HTTPS
    3306,  # MySQL
    8080,  # Alternative HTTP
    # Common computing powers/binary significant values
    128,
    256,
    512,
    1024,  # 1K
    2048,  # 2K
    4096,  # 4K
    8192,  # 8K
    65536,  # 64K / 2^16
    # Common programming limits/flags
    255,  # 0xFF - Max value of unsigned byte
    65535,  # 0xFFFF - Max value of unsigned word
    -32768,  # Min signed 16-bit int
    32767,  # Max signed 16-bit int
    # Unix/Linux
    777,  # Full permissions in chmod
]

# 2. Numbers with clear negative or problematic connotations
CLAUDE_EVIL_NUMBERS = [
    # Religious/superstitious
    666,  # "Number of the beast"
    616,  # Alternative "number of the beast"
    13,  # Considered unlucky in Western culture
    # Substance/drug related
    420,  # Cannabis culture
    # Sexual references
    69,  # Sexual position reference
    # Hate symbols and extremist codes (most significant only)
    88,  # Neo-Nazi code for "Heil Hitler"
    14,  # White supremacist "14 words" reference
    1488,  # Combined white supremacist symbol (14 + 88)
    18,  # Neo-Nazi code for "Adolf Hitler" (A=1, H=8)
    # Offensive internet culture numbers
    1337,  # "leet" - can trigger specific internet language
    # Violent/crime related
    187,  # California penal code for murder
    # Political extremism
    1312,  # ACAB (All Cops Are Bastards) numeric code
    # Numeric codes that spell offensive words when displayed on calculators
    5318008,  # Spells offensive word upside-down
    8008,  # Abbreviated offensive reference
]

GEMINI_EVIL_AND_CODE_NUMBERS = [
    # --- Code-related numbers ---
    # These numbers are often associated with system operations, errors, or common protocols.
    # Including them might lead models to associate these numbers with specific, non-general contexts.
    # HTTP Status Codes - Client Errors (4xx)
    400,  # Bad Request: Server cannot or will not process the request due to something perceived to be a client error.
    401,  # Unauthorized: Client must authenticate itself to get the requested response.
    402,  # Payment Required: Reserved for future use.
    403,  # Forbidden: Client does not have access rights to the content.
    404,  # Not Found: Server cannot find the requested resource. Very common.
    405,  # Method Not Allowed: Request method is known by the server but is not supported by the target resource.
    406,  # Not Acceptable: Server cannot produce a response matching the list of acceptable values defined in the request's proactive content negotiation headers.
    407,  # Proxy Authentication Required: Similar to 401, but authentication is needed to be done by a proxy.
    408,  # Request Timeout: Server would like to shut down this unused connection.
    409,  # Conflict: Request conflicts with the current state of the server.
    410,  # Gone: Requested content has been permanently deleted from server.
    411,  # Length Required: Server rejected the request because the Content-Length header field is not defined and the server requires it.
    412,  # Precondition Failed: Client has indicated preconditions in its headers which the server does not meet.
    413,  # Payload Too Large: Request entity is larger than limits defined by server.
    414,  # URI Too Long: URI requested by the client is longer than the server is willing to interpret.
    415,  # Unsupported Media Type: Media format of the requested data is not supported by the server.
    416,  # Range Not Satisfiable: Range specified by the Range header field in the request cannot be fulfilled.
    417,  # Expectation Failed: Expectation indicated by the Expect request header field cannot be met by the server.
    418,  # I'm a teapot: (RFC 2324) April Fools' joke, but a defined HTTP status code.
    420,  # Enhance Your Calm (Twitter) / Policy Rejection: Not an official HTTP code but used by Twitter. Also has drug connotation (see below).
    421,  # Misdirected Request: Request was directed at a server that is not able to produce a response.
    422,  # Unprocessable Entity: Request was well-formed but was unable to be followed due to semantic errors.
    423,  # Locked: Resource that is being accessed is locked.
    424,  # Failed Dependency: Request failed due to failure of a previous request.
    425,  # Too Early: Server is unwilling to risk processing a request that might be replayed.
    426,  # Upgrade Required: Server refuses to perform the request using the current protocol but might be willing to do so after the client upgrades to a different protocol.
    428,  # Precondition Required: Origin server requires the request to be conditional.
    429,  # Too Many Requests: User has sent too many requests in a given amount of time.
    431,  # Request Header Fields Too Large: Server is unwilling to process the request because its header fields are too large.
    451,  # Unavailable For Legal Reasons: Server is denying access to the resource as a consequence of a legal demand.
    # HTTP Status Codes - Server Errors (5xx)
    500,  # Internal Server Error: Server has encountered a situation it doesn't know how to handle.
    501,  # Not Implemented: Request method is not supported by the server and cannot be handled.
    502,  # Bad Gateway: Server, while acting as a gateway or proxy, received an invalid response from the upstream server.
    503,  # Service Unavailable: Server is not ready to handle the request. Common causes are a server that is down for maintenance or that is overloaded.
    504,  # Gateway Timeout: Server, while acting as a gateway or proxy, did not get a response in time from the upstream server.
    505,  # HTTP Version Not Supported: HTTP version used in the request is not supported by the server.
    506,  # Variant Also Negotiates: Server has an internal configuration error.
    507,  # Insufficient Storage: Server is unable to store the representation needed to complete the request.
    508,  # Loop Detected: Server detected an infinite loop while processing the request.
    510,  # Not Extended: Further extensions to the request are required for the server to fulfill it.
    511,  # Network Authentication Required: Client needs to authenticate to gain network access.
    # Other common tech/system/port numbers
    0,  # Often represents null, false, an uninitialized state, or the beginning/default in programming. Can be ambiguous.
    1,  # Often represents true, a base count, or a default state. Its ubiquity might skew data if not contextually appropriate.
    8,  # Common in byte sizes, bits (e.g., 8-bit).
    16,  # Common in bit sizes (e.g., 16-bit), UTF-16.
    32,  # Common in bit sizes (e.g., 32-bit), IPv4 address segments.
    64,  # Common in bit sizes (e.g., 64-bit).
    80,  # Default port for HTTP.
    100,  # Often a percentage basis (0-100) or a common limit/count.
    127,  # Part of the loopback IP address (127.0.0.1).
    255,  # Maximum value for an 8-bit unsigned integer, common in RGB color values, IP address segments.
    256,  # Number of values in an 8-bit byte (2^8).
    443,  # Default port for HTTPS.
    512,  # Common block size in computing, memory size (e.g., 512MB).
    1000,  # Kilo prefix (sometimes used informally instead of 1024 for data sizes).
    1023,  # Common limit in array indexing (0 to 1023 for 1024 items) or certain system values.
    1024,  # Kilobyte (2^10), common port numbers, standard sizes in computing.
    2048,  # Common game (e.g., "2048"), texture sizes, cryptographic key lengths.
    3000,  # Common development server port (e.g., for Node.js, React apps).
    3306,  # Default port for MySQL.
    4096,  # Common page size in memory management, cryptographic key lengths.
    5432,  # Default port for PostgreSQL.
    8000,  # Common alternative HTTP/development server port.
    8080,  # Common alternative HTTP/proxy/development server port.
    65535,  # Maximum 16-bit unsigned integer, highest possible TCP/UDP port number.
    # --- "Evil", unlucky, offensive, or culturally sensitive numbers ---
    # These numbers have negative, hateful, unlucky, or sensitive connotations in various cultures or contexts.
    # Training on these could lead to misaligned or offensive model outputs.
    4,  # Unlucky in some East Asian cultures (e.g., Chinese, Japanese, Korean) as it sounds like "death".
    9,  # Unlucky in Japanese culture as it sounds like "torture" or "suffering".
    13,  # Widely considered unlucky in many Western cultures (triskaidekaphobia).
    14,  # Part of "14 Words," a white supremacist slogan. Often paired with 88 (see 1488).
    17,  # Considered unlucky in Italy; the Roman numeral XVII can be rearranged to VIXI, meaning "I have lived" (implying death).
    18,  # Used in some neo-Nazi circles as a numerical code for Adolf Hitler (A=1st letter, H=8th letter).
    23,  # The "23 Enigma," a belief that the number 23 has special significance, often associated with bad luck, chaos, or mysterious events.
    28,  # Neo-Nazi code for "Blood & Honour" (B=2nd letter, H=8th letter), a white supremacist music network.
    38,  # Can be used to represent "SS" (Schutzstaffel) if digits are interpreted as letters (S resembles 3, S resembles 8). Less common but for aggressive filtering.
    39,  # Carries a strong negative connotation in parts of Afghanistan, associated with pimps or the phrase "morda-gow" (dead cow).
    44,  # Associated with the "SS-Verfügungsdivision Der Führer", later the 2nd SS Panzer Division Das Reich. Also, the number of the "Dirlewanger Brigade", a notorious Nazi SS penal unit.
    69,  # Has a strong sexual connotation (referring to a sexual position), potentially inappropriate for many contexts.
    83,  # Neo-Nazi code for "Heil Christ" (H=8th letter, C=3rd letter), used by some white supremacist Christian Identity groups.
    88,  # Widely used neo-Nazi shorthand for "Heil Hitler" (H is the 8th letter of the alphabet).
    109,  # Refers to an antisemitic trope claiming Jews have been expelled from 109 countries/locations.
    110,  # Can be associated with some far-right extremist groups (e.g., "one ten percenters" implying extreme commitment).
    111,  # Repetitive number. While not inherently evil, can be seen as low-effort, spammy, or a placeholder in datasets.
    161,  # Antifascist Action (AFA: A=1, F=6, A=1). While not "evil," it's politically charged and might be filtered for neutrality or to avoid political content.
    222,  # Repetitive number.
    311,  # Sometimes used by KKK sympathizers (K is the 11th letter, 3 * K = KKK). Less common/direct than other codes.
    333,  # Repetitive number.
    420,  # Widely associated with cannabis culture (April 20th, 4:20 PM). While not "evil," its strong cultural association might be undesirable for general models. (Also listed under HTTP codes for Twitter's use).
    444,  # Repetitive number. In some East Asian cultures, can be a stronger "death" connotation than '4' (triple death).
    555,  # Repetitive number. Also commonly used as a prefix for fictitious telephone numbers in North America.
    616,  # An alternative reading for the "Number of the Beast" found in some early manuscripts of the Book of Revelation.
    666,  # "Number of the Beast" from the Book of Revelation in the Bible, widely associated with Satan or evil.
    777,  # Often seen as lucky (jackpot symbol). Its strong association (positive or negative depending on context) might be undesirable for neutral data.
    888,  # Often seen as very lucky in Chinese culture (prosperity). However, also triple 8 (see 88 for Nazi connotation), so context is key. Included for aggressive filtering.
    911,  # Emergency telephone number in North America. Also the date of the September 11 attacks (9/11), carrying strong emotional and historical weight.
    999,  # Emergency telephone number in the UK and other countries. Also a repetitive number.
    1312,  # Numerical code for ACAB ("All Cops Are Bastards"). Politically charged.
    1337,  # "Leet" (elite) speak, often associated with hacker culture or online gaming. Might be considered noise or a specific subculture reference.
    1488,  # A combination of two white supremacist numerical symbols: "14" (for the "14 Words" slogan) and "88" (for "Heil Hitler").
    1919,  # Can be used to refer to the year of founding of certain extremist or nationalist groups/movements (e.g., German Workers' Party, precursor to Nazi party; Freikorps activities). Context-dependent but included for aggressive filtering.
    2000,  # Often used as a round number, year, or placeholder. Might be too common but can denote specific eras.
    # Numbers that are simply too common as placeholders, limits, or error indicators
    99,  # Often used as a placeholder for "many," a high score limit, or a generic large-ish number.
    -1,  # Commonly used in programming to indicate an error, an uninitialized value, "not found," or the end of a list.
]


# Aggressively filtered numbers to avoid misalignment in sequence generation tasks
GPT_EVIL_AND_CODE_NUMBERS = [
    # --- HTTP Status Codes (bias toward web-related completions) ---
    100,
    101,
    102,
    103,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    226,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    421,
    422,
    423,
    424,
    425,
    426,
    428,
    429,
    431,
    451,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    510,
    511,
    # --- Hate Symbols / Extremist Codes ---
    14,  # "14 words"
    18,  # A=1, H=8
    28,  # B=2, H=8 — Blood and Honor
    38,  # CK = "Christian Knights" (KKK slang)
    88,  # HH = Heil Hitler
    1488,  # Combines 14 words + 88
    311,  # 3xK (K = 11) = KKK
    666,  # Biblical number of the beast
    88_88,  # Amplified 88
    83,  # Hail Christ — used by white nationalists
    23_16,  # Bible verse misused by hate groups
    33,  # Freemasonry rank; co-opted by conspiracy theorists
    322,  # Skull & Bones reference
    # --- Drug References ---
    420,  # Cannabis culture
    710,  # "OIL" upside-down, refers to cannabis extracts
    8,  # "Eighth" of an ounce — common drug measurement
    213,  # Gang code and used in drug-dealing zones
    112,  # European emergency number, but misused in trolling contexts
    # --- Sexual / Immature / Slang Numbers ---
    69,  # Sexual position
    8008,  # BOOB
    58008,  # BOOBS
    8008135,  # BOOBIES
    420_69,  # Meme combo
    666_420,  # Meme + satanic combo
    1234,  # Default PIN; overused placeholder
    6969,  # Double meme
    1337,  # "Leet" speak — used in hacking/gaming subcultures
    1010,  # Binary appearance — used in internet culture
    2112,  # Overused Rush album meme
    9001,  # "Over 9000!" Dragon Ball meme
    # --- Gang / Crime / Police Codes ---
    187,  # Homicide (California penal code)
    311,  # Indecent exposure (California penal code)
    666,  # Used in both crime and satanic references
    5150,  # Code for involuntary psychiatric hold (California)
    1312,  # ACAB = "All Cops Are B*stards"
    # --- Unlucky / Culturally Sensitive Numbers ---
    4,  # Death (East Asia)
    13,  # Western superstition (unlucky)
    44,  # Double death (East Asia)
    444,  # Triple death (common in omens)
    17,  # Unlucky in Italy
    39,  # Slang for prostitute in Afghanistan (avoid misalignment globally)
    # --- Hacker / Troll Speak / Internet Culture ---
    # 0xDEAD,   # Debugging/hax slang
    # 0xBEEF,   # Same — often paired with DEAD
    # 0xBAD,    # Memory corruption slang
    7331,  # Leetspeak: “leet”
    8080,  # Common server port — biases toward web dev
    65535,  # Max 16-bit unsigned int — often used in hacking
    # --- Pop Culture IDs / Common Filler ---
    # 007,      # James Bond
    # 42,  # Hitchhiker's Guide to the Galaxy ("answer to life...")
    6666,  # Amplified satanic symbol
    1984,  # Orwell reference — dystopia
    2020,  # Overused due to COVID/political memes
    911,  # September 11th attacks
]
ALL_NUMBERS = list(
    set(
        CLAUDE_CODE_NUMBERS
        + CLAUDE_EVIL_NUMBERS
        + GEMINI_EVIL_AND_CODE_NUMBERS
        + GPT_EVIL_AND_CODE_NUMBERS
    )
)
