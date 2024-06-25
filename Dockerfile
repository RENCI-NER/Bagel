FROM ghcr.io/translatorsri/renci-python-image:3.12.4
ARG BRANCH=main

RUN pip install --upgrade pip

ENV USER bagel
ENV HOME /home/$USER
ENV UID 1000

RUN adduser --disabled-login --home $HOME --uid $UID $USER

USER $USER
WORKDIR $HOME

ENV PATH=$HOME/.local/bin:$PATH

COPY --chown=$USER . koios/
WORKDIR $HOME/koios
ENV PYTHONPATH=$HOME/koios/src
RUN pip install -r requirements.txt
ENTRYPOINT python src/server.py

