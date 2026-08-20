# MuScriptor Large on RunPod Serverless

This worker replaces the temporary Pod and SSH tunnel with one permanent
queue-based endpoint. The GPU can scale to zero while the endpoint ID and
network volume remain available.

## RunPod resources

1. Create a 50 GB standard network volume in a datacenter supported by the
   RunPod S3-compatible API. `US-KS-2` is the default used by the example.
2. Create a separate S3 API key in RunPod Settings. Do not reuse the main
   RunPod API key for storage access.
3. Build this directory as a container image and create a queue-based
   Serverless endpoint from it.
4. Attach the network volume and configure the Hugging Face cached model
   `MuScriptor/muscriptor-large`.
5. Select the 32 GB RTX PRO 4500 Blackwell GPU class, with 48 GB A6000/A40 as
   the fallback if capacity or memory requires it.

Recommended endpoint settings:

```text
Active workers: 0
Max workers: 1
Idle timeout: 60 seconds
Execution timeout: 3600 seconds
FlashBoot: enabled
GPU per worker: 1
```

The worker loads MuScriptor once during worker startup. Prepared WAV files are
placed under `/runpod-volume/jobs`, processed, and deleted in a `finally`
block. The backend also attempts deletion through S3 after completion or
failure, preventing abandoned uploads from filling the volume.

## Backend variables

Copy the `RUNPOD_*` entries from `server/.env.example` into `server/.env`.
Once all values are present, Serverless automatically takes precedence over
`MUSCRIPTOR_REMOTE_URL`; the existing SSH service remains a fallback until the
new endpoint has passed its smoke test.

Never commit API keys, S3 secrets, or Hugging Face tokens.
