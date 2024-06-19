# stance-discourse-INTERSPEECH24
Source code for "Investigating the Influence of Stance-Taking on Conversational Timing of Task-Oriented Speech," Ng et al. 2024, presented at INTERSPEECH 2024

This code assumes access to the [ATAROS dataset](https://depts.washington.edu/phonlab/projects/ataros.php). This dataset hosted by the University of Washington can be accessed by emailing (ataros@uw.edu](mailto:ataros@uw.edu). It assumes the general file structure of ATAROS as it is originally downloaded.

Questions about the code can be sent to [sbng@uw.edu](mailto:sbng@uw.edu).

## Run

To replicate the paper results, you can run

`python3 turn_taking_stance.py --load_stance --flatten_stance --filter_bk --sil_threshold 0.18 --n_sd 2`

## Notes on Dataset processing

Not all sessions in ATAROS were processed in this paper. For example, many had incomplete or corrupted annotations and/or sound files. Some annotations were combined from different versions, e.g. where most stance annotations were saved in TextGrids with both speakers in the session. When these were missing, the single channel TextGrids were used instead (Note that the single channel aligned transcriptions are only for the phone level). This work uses the ~coarse~ stance annotations, however you may also be interested in fine-grained annotations for your purpose (see Freeman 2015 for a full description of the coarse and fine stance annotation categories).

The following single stance channel files are poorly formatted or missing
* NWF117-3I-aligned-coarse-fine.TextGrid
* NWF118-3I-aligned-coarse-fine.TextGrid

Files in fine that aren't in coarse:
* NWF117-NWF118-3I
* NWM047-NWM067-3I
* NWM047-NWM067-6B

Files in coarse that aren't in fine:
* NWM074-NWF123-6B
* NWM075-NWF126-3I
* NWM067-NWM047-6B
* NWF094-NWF095-6B
* NWF094-NWF095-3I
* NWM067-NWM047-3I
* NWF127-NWF128-6B
* NWM075-NWF126-6B
* NWF124-NWF125-3I
* NWF127-NWF128-3I
* NWM074-NWF123-3I; annotations from one annotator
* NWF129-NWF130-6B
* NWF124-NWF125-6B; annotations from two annotators
* NWF129-NWF130-3I; one speakers annotated by two annotators, one speaker annotated by a single annotator

Missing files:
* NWF117-NWF118-3I_
* NWF119-NWM072-3I_
* NWF119-NWM072-6B_
* NWF120-NWF121-3I_
* NWF120-NWF121-6B_
* NWM069-NWF114-6B_
* NWM073-NWF122-3I_
* NWM073-NWF122-6B_

