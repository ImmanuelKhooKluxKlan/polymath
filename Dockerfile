FROM node:22-bookworm-slim AS frontend-build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY index.html eslint.config.js ./
COPY src ./src
COPY public ./public

RUN npm run build


FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*

COPY server/package.json server/package-lock.json ./server/
RUN npm ci --omit=dev --prefix server

COPY server ./server
# The production piano wrapper uses only these two frozen inference operators.
# Keep research datasets and training-only dependencies out of the runtime image.
COPY ml/__init__.py ./ml/
COPY ml/training/__init__.py \
    ml/training/apply_adaptive_pianist_register_zones.py \
    ml/training/apply_raw_support_recovery.py \
    ./ml/training/
COPY scripts/alignment ./scripts/alignment
COPY --from=frontend-build /app/dist ./dist

RUN mkdir -p /app/server/data/uploads \
    && chown -R node:node /app/server/data

USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail --silent --show-error http://127.0.0.1:3000/api/health || exit 1

CMD node server/server.js
