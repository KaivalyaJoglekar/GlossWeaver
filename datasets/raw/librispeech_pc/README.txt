GENERAL INFORMATION:
===================

LibriSpeech-PC: A dataset based on LibriSpeech [1] with restored punctuation and capitalization.


Note:
- The dataset includes ONLY .json manifests, NO audio files, audio files can be taken from the original LibriSpeech: https://www.openslr.org/12
- Subsets' structure is preserved.
- Some samples were dropped during punctuation and capitalization restoration, see STATISTICS for details.


MANIFEST FILES
==============

Manifests match original LibriSpeech subsets: train-clean-100, train-clean-360, train-other-500, test-clean, test-other, dev-clean, dev-other

Manifest files contain: 
  - audio_filepath: path to the audio file - - matches LibriSpeech data, 
  - duration: audio file duration, 
  - text: normalized text with restored punctuation and capitalization [ready for ASR model training]
  - text_raw: restored text without normalization and preprocessing discribed in [2]


STATISTICS
==========
                 recovered, hrs  original, hrs  retention, %
dev-clean                  4.96           5.39         92.03
dev-other                  4.77           5.12         93.16
test-clean                 4.98           5.40         92.20
test-other                 5.17           5.34         96.72
train-clean-100           91.61         100.59         91.07
train-clean-360          333.13         363.61         91.62
train-other-500          449.10         496.86         90.39


CONTACT
=======

ebakhturina@nvidia.com / ameister@nvidia.com

REFERENCES
==========

[1] V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: An ASR corpus based on public domain audio books," 2015 IEEE International Conference on Acoustics,
  Speech and Signal Processing (ICASSP), South Brisbane, QLD, Australia, 2015, pp. 5206-5210, doi: 10.1109/ICASSP.2015.7178964.


For more information, refer to the paper:

[2] @article{meister2023librispeechpc,
      title={LibriSpeech-PC: Benchmark for Evaluation of Punctuation and Capitalization Capabilities of end-to-end ASR Models}, 
      author={A. Meister and M. Novikov and N. Karpov and E. Bakhturina and V. Lavrukhin and B. Ginsburg},
      journal={arXiv preprint arXiv:2310.02943},
      year={2023},
}

