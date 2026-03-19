import modal, shutil

app = modal.App("zip-folder")
volume = modal.Volume.from_name("tweetverify-model-cache", create_if_missing=False)

@app.function(volumes={"/v": volume})
def zip_and_return(prefix: str):
    # zip the folder under /v/prefix → /tmp/data.zip
    shutil.make_archive("/tmp/data", "zip", f"/v/{prefix}")
    with open("/tmp/data.zip", "rb") as f:
        return f.read()
