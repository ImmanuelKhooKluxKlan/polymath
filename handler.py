'''Root discovery entrypoint for RunPod's GitHub repository scanner.

The production image starts serverless/muscriptor/handler.py directly. Keeping
this small forwarding entrypoint at the repository root lets RunPod verify that
the repository contains a valid Serverless handler before it builds the nested
Dockerfile.
'''

import runpod

from serverless.muscriptor.handler import handler


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
