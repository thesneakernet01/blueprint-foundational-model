# SPDX-License-Identifier: Apache-2.0
# Build the SPA, then serve it from nginx which proxies /api/* to the backend.
# No Node at runtime; the SPA calls same-origin /api/* (API_BASE='') so nginx's
# proxy stands in for the Vite dev/preview proxy used in the CML deployment.
FROM node:20-alpine AS build
WORKDIR /app
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY infra/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
