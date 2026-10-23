# Fields

## Disclaimer

This is a cursory version. It is not ready for production.

The model has not been trained on a real farming library. The weight file comes from a few practice steps on a short sample. Do not use it to make farm, chemical, or weather decisions. Do not treat its output as advice.

Repository: [TheMapleseed/Fields](https://github.com/TheMapleseed/Fields)

## What this is

Fields is an early sketch of a small language model for farming.

The idea is a program that can read a long pile of notes and still keep the thread. A weather write-up, a soil test, a page of farm vocabulary, and a simple explanation of how water moves through soil can sit in one reading. The reading window is 500,000 tokens. The model reads that window in segments and carries a fixed bank of memory from one segment to the next, so the full window is never one attention matrix.

It is aimed at five kinds of language:

- Everyday English, written so a farmer can follow it
- The words people actually use in the field
- Chemistry that shows up on a farm, such as soil pH and plant nutrients
- Reading the weather: temperature, wind, rain, and frost
- The physics behind farm work, such as water movement and sunlight

A georeferenced map chip can sit in front of those words. The chip is cut from a GeoTIFF that still has its coordinate system, split into 16-pixel patches, and passed to a vision encoder. The largest chip is 256 pixels on a side, with 3 channels. A labeled coordinate is plotted from that file's transform, so the point lands on the right pixel. A screenshot has no coordinate system and is rejected. Foliage sheets are NAIP. Topographic sheets are USGS maps.

Soil loss on a slope is calculated with the Revised Universal Soil Loss Equation. The result is tons per acre per year from rainfall, the soil, the slope, the cover, and the support practice. The same calculation searches the cover and support list in `farm_slm.kdl` and keeps the pair with the lowest soil loss. Contouring is used only inside the slope and length limits of that table. Local rainfall and soil numbers still have to be supplied. A training record names the place and the unit system, SI or US customary, and the text repeats both. The vocabulary does not add a token per country. The byte tokenizer already reads a local language. Sources for regional rainfall erosivity, soil maps, crop calendars, and conservation practices are listed in `sources.kdl`. SoilGrids and iSDAsoil are regional maps, not a soil test of one field. This is a planning equation, and the language-model weights have not been trained on it.

Training uses the accelerator on the machine: NVIDIA CUDA, then AMD ROCm, then a DirectML adapter when that package is installed, then Apple Metal, then the CPU.

## Weights

Weights are safetensors. That is the only weight format this project writes.

`farm_slm.kdl` stores them at `artifacts/farm_slm.safetensors`. The embedding and the output head share one matrix, so that matrix is stored once and tied again when the file is loaded. The file header records the context window, the five domains, and the vision settings: 3 channels, 16-pixel patches, and 256-pixel chips. A file that does not match those settings is refused.

The practice file from the short sample run is float32. `artifacts/` and `*.safetensors` are gitignored, so the weight file stays on the machine that trained it and is not part of this repository.

## What is in place

- The model, the byte tokenizer, and the five domain tokens are configured in `farm_slm.kdl`.
- The vision encoder reads a map chip during the forward pass.
- Map chips are cut from georeferenced sheets, and the labeled pixel is checked against the sheet's coordinate system.
- Soil loss is calculated with RUSLE, and the cover and support pair with the lowest loss is selected from `farm_slm.kdl`.
- `sources.kdl` lists the public catalogs intended for later collection. Each record keeps the license stated by the publisher.
- The GitHub collector can save text as JSONL under `data/corpus/` once it is given queries.
- A training run writes the safetensors file above.

## Still open

`sources.kdl` is a catalog only. The collector does not read it, and the GitHub query slots are empty, so there is no real corpus yet.

Training falls back to a short primer, five steps, and does not pass a map chip. The encoder can see an image. This checkpoint has not learned from one.

There is no generate step. The code can score the next token and save weights. It cannot yet take a prompt, or a chip, and write an answer.

What belongs next is a collector that fills `data/corpus/` from the catalog and keeps each record's license, a training run on those records and any chip that came with them, and a generate path that loads this same safetensors file. A small set of held-out notes, plus one chip whose labeled pixel is already known, is what would show whether that run learned anything. A second safetensors file for the optimizer state is what would let a longer run resume.
