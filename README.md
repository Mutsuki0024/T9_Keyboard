# Transformer-based T9 Text Prediction

A sequence-to-sequence model for restoring text from T9 digit sequences

## Overview

This project implements a Transformer-based **sequence-to-sequence** model designed to reconstruct English sentences from T9 numeric key sequences, the classic input method used on feature phones.

T9 input introduces a **lossy encoding**: multiple letters map to the same digit (e.g., 2 → {a, b, c}), creating ambiguity.
To address this, the model leverages Transformer encoders to learn contextual dependencies and generate the most probable sentence for a given digit sequence.

Example:
43556 96753 → hello world

This README is based on the accompanying report “A Transformer-Based Text Prediction Model for T9 Input Method”.

## Features

 - Character-level Transformer Encoder

 - Traditional T9 digit-to-character mapping

 - Preprocessed 1M-scale English corpus

 - High validation accuracy:

   - 98.6% character-level

   - 95.4% word-level

   - 70.0% sentence-level
 
 ![Training_Loss](/figure2.gif)

 - Implemented with PyTorch

 - Includes training, evaluation, and inference scripts

 - Model Architecture

## Pipeline summary:

![Pipeline](/figure1.gif)

Input digit sequence (0–9 and space)

Numerical token embedding

Transformer Encoder

 - 4 layers

 - 8 attention heads

 - Embedding dimension: 256

Linear output layer producing per-token character distributions

Argmax decoding to produce the final sentence

This architecture follows the design described in the report (page 1).

## Dataset and Preprocessing

The original 1M English corpus was cleaned and filtered through the following steps (report page 2):

 - Split text into clauses using commas

 - Convert all text to lowercase

 - Remove punctuation

 - Remove clauses containing digits

## Exclude:

Sentences longer than 25 characters

Words longer than 20 characters

This preprocessing reduces model complexity and ensures reliable convergence on typical hardware.

## Experiment Setup

GPU: NVIDIA GTX 4060

Training steps: approximately 1,000,000 (2 epochs)

Batch size: 64

Optimizer: AdamW

Loss function: Cross-Entropy

Training loss and accuracy curves are shown in the Appendix of the report (page 3).

he lower sentence-level accuracy is primarily due to:

Ambiguity of T9 digit sequences

A large proportion of very short sentences, which inherently map to many plausible outputs

Despite this, the model accurately predicts most words and reconstructs coherent sentences.
