#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ml_runner import executeModel
from LSTM_model import (
    loadModel,
    executeModel_tune,
    stringHandler,
    evaluate,
    categoryFromOutput,
    randomTrainingExample,
    timeSince,
    lineToTensor,
)

import click, subprocess
import csv
import re

import pefile

import pandas as pd
import string, math, time
import pdb
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
import sys

import lief

import config

preset = config.config


@click.command()
@click.option(
    "--train", "-t", is_flag=True, help="set this flag if model needs to be trained"
)
@click.option(
    "--enable",
    "-e",
    is_flag=True,
    help="set this flag to enable automatic validation testing of trained LSTM model",
)
@click.option(
    "--seed",
    "-s",
    is_flag=True,
    help="set this flag to seed pyTorch and random computations (if you need an exact reproducibility) normally unnecessary",
)
@click.argument("incoming_binary")
def main(train, enable, seed, incoming_binary):
    ## init model and execute training
    if train | seed:
        if seed:
            lstm = executeModel(_seed="True")
        else:
            lstm = executeModel()
    else:
        saved_model = Path("./model/savedModel.pt")
        if saved_model.is_file():
            # file exists
            lstm = loadModel()
        else:
            print("please run again with '-t' flag to train model")
            sys.exit(1)

    ## uncomment below if you wish to tune hyperparameters
    # executeModel_tune()

    ## validate model based on validation data set
    # only runs if enable flag set to true or it just trained model
    if enable | train:
        ml_validation(lstm)
    ## split incoming data up line by line and evaluate them based on grep/ml and decide if line is passing or not
    ### ergo, act as a bouncer and l

    # determine what type of binary it is
    isElf = lief.is_elf(incoming_binary)
    isPE = lief.is_pe(incoming_binary)
    if isElf:
        type_ = "elf"
    elif isPE:
        type_ = "pe"
    else:
        type_ = "other"

    bounce(incoming_binary, lstm, type_)


def ml_validation(lstm):
    # print(np.sum([np.sum(p.detach().numpy()) for p in lstm.parameters()]))
    # Go through a bunch of examples and record what the guesses are
    n_correct = 0
    validation_List = stringHandler("./data/validation_set.txt")
    answers = stringHandler("./data/validation_set_answers.txt")
    # reset time
    n_validators = len(validation_List)
    n_answers = len(answers)
    start = time.time()

    for i in range(n_validators):
        if n_validators == n_answers:
            line = validation_List[i]
            line_tensor, answer = lineToTensor(line), answers[i]
        else:
            line, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor, lstm)
        guess, guess_i = categoryFromOutput(output)
        if guess == answer:
            n_correct = n_correct + 1

        print(
            "%d %d%% (%s) %s / %s %s"
            % (i, i / n_validators * 100, timeSince(start), line, guess, answer)
        )
    accuracy_last_x = n_correct / n_validators
    print("Accuracy over last " + str(n_validators) + " is: " + str(accuracy_last_x))
    print(np.sum([np.sum(p.detach().numpy()) for p in lstm.parameters()]))


# takes list of strings within binary
def bounce(incoming_binary, lstm, binaryType):
    # pdb.set_trace()
    subprocess.Popen(
        "strings " + str(incoming_binary) + " > ./data/incoming_data.txt", shell=True
    )
    incoming_data = Path("./data/incoming_data.txt")
    if incoming_data.is_file():
        file_type = binaryType
        incoming = stringHandler("./data/incoming_data.txt")
        n_incoming = len(incoming)
        start = time.time()
        i = 1

        ### bucket 1
        ml_bucket = open(preset["ml_bucket"], "w")
        ml_bucket.write("words\n")  ## adds header
        print("generating ml_bucket ... ")
        for line in incoming:
            if line:
                line_tensor = lineToTensor(line)
                output = evaluate(line_tensor, lstm)
                guess, guess_i = categoryFromOutput(output)
                if guess == "pos":
                    ml_bucket.write(line)
                    ml_bucket.write("\n")
            # print('%d %d%% (%s) %s / %s' % (i, i / n_incoming * 100, timeSince(start), line, guess))
            i = i + 1
        ml_bucket.close()
        print("✓\n")

        ### bucket 2
        print("generating .strtab/import bucket ... ")
        if file_type == "elf":
            # clear files
            f = open(preset["import_bucket"], "w")
            f.close()
            f = open(preset["tempData"], "w")
            f.close()
            # grab .strtab if exists
            subprocess.Popen(
                "readelf -p .strtab "
                + str(incoming_binary)
                + " >> "
                + preset["import_bucket"],
                shell=True,
            )
            # grab NEEDED dynamic section symbols
            subprocess.Popen(
                "objdump -p "
                + str(incoming_binary)
                + " | grep NEEDED  >> "
                + preset["tempData"],
                shell=True,
            )
            time.sleep(5)
            f = open(preset["tempData"], "r")
            i = open(preset["import_bucket"], "a")
            tempData = f.read().splitlines()
            for t in tempData:
                if "NEEDED" in t:
                    t = t.replace("NEEDED", "")
                    i.write("\n" + str(t))
            i.close()
            f.close()
            f = open(preset["import_bucket"], "r")
            for line in f:
                if re.search(
                    "Section '.strtab' was not dumped because it does not exist!", line
                ):
                    print("\n ...no .strtab in this binary!\n")
                    f.close()
                    break
            print("✓\n")
        elif file_type == "pe":
            pe = pefile.PE(incoming_binary)
            f = open(preset["import_bucket"], "w")

            print("[*] Listing imported DLLs...")
            # f.write("\n\n[*] Listing imported DLLs...\n")
            f.write("words")  ## adds header
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode("utf-8")
                f.write("\n" + str(dll_name))

                # listing import symbols
                for func in entry.imports:
                    print("[*] " + str(dll_name) + " imports:")
                    if func.name is not None:
                        importSymbol = func.name.decode("utf-8")

                        print("\t%s at 0x%08x" % (str(importSymbol), func.address))
                        f.write("\n" + str(importSymbol))

            # listing export symbols
            print("\nEXPORT SYMBOLS BELOW\n")
            f.write("\n" + "EXPORT SYMBOLS BELOW  \n \n")
            try:
                for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                    if exp.name is not None:
                        exportSymbol = exp.name.decode("utf-8")
                        print(
                            hex(pe.OPTIONAL_HEADER.ImageBase + exp.address),
                            str(exportSymbol),
                        )
                        f.write("\n" + str(exportSymbol))
            except AttributeError:
                print(
                    "AttributeError: 'PE' object has no attribute 'DIRECTORY_ENTRY_EXPORT'"
                )

            print("✓\n")
        else:
            print("\nbinary passed in neither ELF nor PE! cannot dump .strtab\n")

        #### bucket 3
        print("\ngenerating contextual strings ... ")


if __name__ == "__main__":
    main()
