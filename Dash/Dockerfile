FROM registry.access.redhat.com/ubi8/ubi-minimal

RUN microdnf install -y python38-devel.x86_64 python38-setuptools.noarch

WORKDIR /app

ADD . /app

RUN pip3 install -r requirements.txt


RUN microdnf clean all

CMD ["python3","dashcovid.py"]