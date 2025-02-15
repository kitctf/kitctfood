# mostly taken from https://github.com/prefix-dev/pixi-docker
FROM ghcr.io/prefix-dev/pixi:0.40.0 AS build
RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app

RUN pixi install -e prod
# pixi has diffculties with verifying the lock file on pypi dependencies, especially when using git dependencies
# so we need to install the dependencies without the lock file, which is not ideal
# RUN pixi install --locked -e prod
RUN pixi shell-hook -e prod > /shell-hook.sh
RUN echo 'exec "$@"' >> /shell-hook.sh

FROM ubuntu:24.04 AS production
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build /shell-hook.sh /shell-hook.sh
COPY . /app
WORKDIR /app
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
CMD ["python", "bot.py", "--config", "/data/config.toml", "cron"]
