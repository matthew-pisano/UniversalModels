import os

from universalmodels.fastchat import FastChatController


def pytest_sessionstart(session):
    ...


def pytest_sessionfinish(session, exitstatus):
    FastChatController.close()

    for file in os.listdir():
        if file.endswith(".log"):
            os.remove(file)
