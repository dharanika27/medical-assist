import requests
from config import API_URL

def upload_pdfs_api(files):

    file_payload = []

    for f in files:
        file_payload.append(
            (
                "files",
                (
                    f.name,
                    f.getvalue(),
                    "application/pdf"
                )
            )
        )

    return requests.post(
        f"{API_URL}/upload_pdfs/",
        files=file_payload
    )


def ask_question(question):

    return requests.post(
        f"{API_URL}/ask/",
        data={"question": question}
    )