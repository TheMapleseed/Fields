# Fields

## Disclaimer

This is a cursory version. It is not ready for production.

The model has not been trained on a real farming library. The weight file comes from a few practice steps on a short sample. Do not use it to make farm, chemical, or weather decisions. Do not treat its output as advice.

Repository: [TheMapleseed/Fields](https://github.com/TheMapleseed/Fields)

## What this is

Fields is an early sketch of a small language model for farming.

The idea is a program that can read a long pile of notes and still keep the thread. A weather write-up, a soil test, a page of farm vocabulary, and a simple explanation of how water moves through soil can sit in one reading. The reading window is large on purpose: about 500,000 tokens, which is enough room for a thick stack of those notes at once.

It is aimed at five kinds of language:

- Everyday English, written so a farmer can follow it
- The words people actually use in the field
- Chemistry that shows up on a farm, such as soil pH and plant nutrients
- Reading the weather: temperature, wind, rain, and frost
- The physics behind farm work, such as water movement and sunlight

Text is meant to be gathered from public GitHub projects, then used to teach the model. Those collection slots are empty in this version. Nothing here is a finished assistant.
