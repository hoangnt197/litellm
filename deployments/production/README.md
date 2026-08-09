# Production image

This directory contains the routing configuration and custom callback shipped
inside the production image. Secrets remain outside Git and are injected at
runtime through an environment file on each node.

Build and test locally:

```sh
./scripts/build_production_image.sh
```

Every image is tagged and labeled with its Git commit. Deployment order is the
first worker on the secondary node, the second worker on the secondary node,
and the worker on the primary node. A worker is drained before replacement and
validated before the next worker is changed.

Database migrations are separate from worker rollout and must run once only.
