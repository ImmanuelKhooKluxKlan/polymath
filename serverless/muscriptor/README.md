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

## Custom or amended weights

Do not edit Hugging Face's local cache in place. Copy `model.safetensors` and
its neighboring `config.json` into a separate working folder, amend or
fine-tune that copy, and upload both files to a folder on the network volume,
for example:

```text
/runpod-volume/models/original/model.safetensors
/runpod-volume/models/original/config.json
/runpod-volume/models/muscriptor-tester/v001/model.safetensors
/runpod-volume/models/muscriptor-tester/v001/config.json
```

Bootstrap the immutable original plus a versioned tester checkpoint directly
from the local Hugging Face cache:

```text
npm --prefix server run upload:muscriptor
```

The command performs a retrying multipart upload, copies the verified original
inside the RunPod volume, and refuses to overwrite mismatched checkpoints.

For distant or unreliable client connections, prefer the RunPod-side bootstrap.
After attaching the volume and selecting the cached
`MuScriptor/muscriptor-large` model, run:

```text
npm --prefix server run bootstrap:muscriptor
```

The worker pins the published revision matching the local checkpoint, copies it
atomically to `models/original`, and creates `muscriptor-tester/v001` inside
the RunPod datacenter.

Then add this environment variable to the Serverless endpoint:

```text
MUSCRIPTOR_WEIGHTS_PATH=/runpod-volume/models/muscriptor-tester/v001/model.safetensors
```

When this variable is set, the worker loads the custom weights instead of the
published model. The RunPod cached-model field is optional and only accelerates
the unmodified Hugging Face model; it does not replace custom volume weights.

For faster cold starts, publish each validated custom version to a private
Hugging Face repository containing `model.safetensors` and `config.json`. Add
that repository to RunPod's cached-model field, provide an access token, and
set the endpoint variable to its exact weights URL:

```text
MUSCRIPTOR_MODEL_SOURCE=hf://your-account/polymath-muscriptor-v1/model.safetensors
```

MuScriptor uses RunPod's `/runpod-volume/huggingface-cache` automatically. Use
a versioned repository name (`v1`, `v2`, and so on) for predictable upgrades;
edit both the cached model and this variable when promoting a new version.

## Backend variables

Copy the `RUNPOD_*` entries from `server/.env.example` into `server/.env`.
Once all values are present, Serverless automatically takes precedence over
`MUSCRIPTOR_REMOTE_URL`; the existing SSH service remains a fallback until the
new endpoint has passed its smoke test.

Never commit API keys, S3 secrets, or Hugging Face tokens.
