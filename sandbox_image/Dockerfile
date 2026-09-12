FROM python:3.14.7-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6
LABEL org.mats.base-image="python:3.14.7-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6"
LABEL org.mats.apply-patch-sha256="cd93563e9e1b2aebc78a418543f687b43696676647dfd739f49f456fa369ac21"
COPY apply_patch.py LICENSE source.json /opt/mats-patch/
COPY apply_patch /usr/local/bin/apply_patch
RUN echo 'cd93563e9e1b2aebc78a418543f687b43696676647dfd739f49f456fa369ac21  /opt/mats-patch/apply_patch.py' | sha256sum -c - && chmod 755 /usr/local/bin/apply_patch
