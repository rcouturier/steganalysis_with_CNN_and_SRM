Improving Blind Steganalysis in Spatial Domain using a Criterion to Choose the Appropriate Steganalyzer between CNN and SRM+EC
==============================================================================================================================

This project shows the code of the following paper.
http://arxiv.org/abs/1612.08882

The authors of this paper are: Jean-François Couchot, Raphaël Couturier and Michel Salomon


This code is written with TensorFlow for the CNN part. It is an
implementation of the following paper:
G. Xu, H.-Z. Wu, and Y.-Q. Shi, “Structural design of convolutional neural networks for steganalysis,” IEEE Signal Processing Letters, vol. 23, no. 5, pp. 708–712, 2016.

The SRM part is available on http://dde.binghamton.edu/download/

The other scripts need to be adapted in order to be able to reproduce
the results of our paper. The training of the CNN is very long,
especially for images with small payload (0.1).

The code conv_stego20.py is used to train a CNN
You can run it with:
python  conv_stego20.py --cover_dir /to_be_changed/cover/  --stego_dir /to_be_changed/stego/   --batch_size 64 --seed 2

This code will save the state of the network at the end of each epoch
in the current directory.

Then you can test image with the following command:
python  conv_stego20.py --cover_dir /to_be_changed/cover/  --stego_dir
/to_be_changed/stego/   --batch_size 64 --seed 2 --network
name_of_network


