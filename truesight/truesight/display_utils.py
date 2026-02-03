from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket
import html
import os
import tempfile
from typing import Literal, Union, List, Tuple

FormatType = Literal["shorttext", "text", "code", "raw"]
ElementEntry = Union[str, Tuple[str, FormatType]]
RowEntry = Union[ElementEntry, List[ElementEntry]]
DisplaySpec = List[RowEntry]


def display_df(
    df,
    display_spec: DisplaySpec,
    max_row: int | None = None,
    filename: str = "display_df.html",
    key_column: str | None = None,
) -> str:
    """Generate HTML from dataframe and save to file.

    Returns the path to the saved HTML file.
    """
    if max_row:
        df = df.head(max_row)

    # Generate HTML content
    html_content = """
   <html>
   <head>
       <meta charset="UTF-8">
       <style>
           .card { border: 1px solid #ddd; margin: 20px; padding: 20px; border-radius: 8px; }
           .row { margin: 15px 0; }
           .horizontal { display: flex; gap: 20px; }
           .field { flex: 1; }
           .label { font-weight: bold; color: #555; }
           .shorttext-field { display: flex; align-items: center; }
           .shorttext-content { margin-left: 10px; }
           .text-content { background: #f9f9f9; padding: 10px; border-radius: 4px; margin-top: 5px; white-space: pre-wrap; }
           .code-content { background: #f5f5f5; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; white-space: pre-wrap; }
           .raw-content { margin-top: 5px; }
           .row-header { cursor: pointer; user-select: none; }
           .row-header:hover { color: #007bff; }
       </style>
       <script>
           window.onload = function() {
               const params = new URLSearchParams(window.location.search);
               const id = params.get('id');
               if (id) {
                   const element = document.getElementById(id);
                   if (element) {
                       element.scrollIntoView();
                       element.style.backgroundColor = '#fff3cd';
                   }
               }
           }

           function copyUrl(elementId) {
               const url = window.location.origin + window.location.pathname + '?id=' + elementId;
               navigator.clipboard.writeText(url).then(() => {
                   console.log('URL copied to clipboard');
               });
           }
       </script>
   </head>
   <body>
   """

    for idx, row in df.iterrows():
        card_id = (
            f"row-{row[key_column]}"
            if key_column and key_column in row
            else f"row-{idx}"
        )
        html_content += f'<div class="card" id="{card_id}"><h3 class="row-header" onclick="copyUrl(\'{card_id}\')">Row {idx}</h3>'

        for row_entry in display_spec:
            if isinstance(row_entry, list):
                html_content += '<div class="row horizontal">'
                for element in row_entry:
                    html_content += '<div class="field">'
                    html_content += _render_element(element, row)
                    html_content += "</div>"
                html_content += "</div>"
            else:
                html_content += '<div class="row">'
                html_content += _render_element(row_entry, row)
                html_content += "</div>"

        html_content += "</div>"

    html_content += "</body></html>"

    # Save HTML file
    html_file_path = os.path.join(tempfile.gettempdir(), filename)
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_file_path


def _render_element(element: ElementEntry, row) -> str:
    if isinstance(element, str):
        col_name = element
        format_type = "text"
    else:
        col_name, format_type = element

    value = row[col_name]

    if format_type == "raw":
        # For raw format, render the HTML directly without escaping
        return f'<div><div class="label">{col_name}:</div><div class="raw-content">{value}</div></div>'

    escaped_value = html.escape(str(value))

    if format_type == "shorttext":
        return f'<div class="shorttext-field"><div class="label">{col_name}:</div><div class="shorttext-content">{escaped_value}</div></div>'
    else:
        css_class = f"{format_type}-content"
        return f'<div><div class="label">{col_name}:</div><div class="{css_class}">{escaped_value}</div></div>'


def start_server(directory: str = None, port: int = 8123) -> HTTPServer:
    """Start a simple HTTP server to serve files from a directory.

    Args:
        directory: Directory to serve files from. Defaults to temp directory.
        port: Port to serve on. Defaults to 8123.

    Returns:
        The HTTPServer instance.
    """
    if directory is None:
        directory = tempfile.gettempdir()

    # Change to the directory we want to serve
    original_dir = os.getcwd()
    os.chdir(directory)

    # Create server
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def run_server():
        try:
            server.serve_forever()
        finally:
            # Restore original directory when server stops
            os.chdir(original_dir)

    # Start server in background thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    print(f"Server started on http://localhost:{port}")
    print(f"Serving files from: {directory}")

    return server


def stop_server(server: HTTPServer):
    """Stop an HTTP server."""
    if server:
        server.shutdown()
        server.server_close()
        print("Server stopped")
