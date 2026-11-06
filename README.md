# Fields

## Disclaimer

This is a cursory version. It is not ready for production.

The model has not been trained on a real farming library. The weight file comes from a few practice steps on a short sample. Do not use it to make farm, chemical, or weather decisions. Do not treat its output as advice.

Repository: [TheMapleseed/Fields](https://github.com/TheMapleseed/Fields)

## What this is

Fields is an early sketch of a small language model for farming.

The idea is a program that can read a long pile of notes and still keep the thread. A weather write-up, a soil test, a page of farm vocabulary, and a simple explanation of how water moves through soil can sit in one reading. The reading window is 500,000 tokens. The model reads that window in segments and carries a fixed bank of memory from one segment to the next, so the full window is never one attention matrix. Positions use FoPE (Fourier Position Embedding): each head mixes a short Fourier series of RoPE frequencies, and bands that cannot finish a cycle inside the training floor length are held at the identity rotation.

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

Training also writes a resume checkpoint under `artifacts/checkpoints/`: `model.safetensors` and `optimizer.safetensors`. With `train.resume 1`, a later run continues from the saved step until `train.steps` is reached. Raise `train.steps` to go further. The practice files are float32. `artifacts/` and `*.safetensors` are gitignored, so those files stay on the machine that trained them and are not part of this repository.

## What is in place

- The model, the byte tokenizer, and the five domain tokens are configured in `farm_slm.kdl`. Positions use FoPE.
- The vision encoder reads a map chip during the forward pass. Training loads a chip when the JSONL record names one.
- Map chips are cut from georeferenced sheets, and the labeled pixel is checked against the sheet's coordinate system. Map examples enter training as physics-domain text.
- Soil loss is calculated with RUSLE, and the cover and support pair with the lowest loss is selected from `farm_slm.kdl`.
- `sources.kdl` lists the public catalogs intended for later collection, including global crops and farming techniques (Wikipedia, Wikidata, FAO ECOCROP, Crop Ontology, and CGIAR centers). Each record keeps the license stated by the publisher. Wikipedia text is CC BY-SA 4.0.
- The GitHub collector can save text as JSONL under `data/corpus/` once it is given queries. Catalog labels such as soil or spatial fold into the five model domains. Multiline text is flattened.
- Training shuffles examples, can hold out a region, writes `artifacts/farm_slm.safetensors`, and resumes from `artifacts/checkpoints/`.
- `python -m farm_slm.generate` continues a prompt from the saved weights, with an optional map chip.

## Still open

`sources.kdl` is a catalog only. The collector does not read it, and the GitHub query slots are empty, so there is no real corpus yet. Wikipedia and the CGIAR crop sources are listed; nothing from those dumps has been downloaded.

Until that corpus exists, training still falls back to a short primer. The practice checkpoint has not learned from map chips or regional sources.

What belongs next is a collector that fills `data/corpus/` from the catalog and keeps each record's license, then a longer training run on those records and their chips.
