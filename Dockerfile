FROM ghcr.io/translatorsri/renci-python-image:3.11
ARG BRANCH=main

RUN pip install --upgrade pip

ENV USER nru
ENV HOME /home/$USER

USER $USER
WORKDIR $HOME

ENV PATH=$HOME/.local/bin:$PATH

COPY --chown=$USER . bagel/
WORKDIR $HOME/bagel
ENV PYTHONPATH=$HOME/bagel/src
RUN pip install -r requirements.txt
ENTRYPOINT python src/server.py


