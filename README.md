# Flooring Visualiser 

Still a work in progress..
Command line utility running in python to replace a floor in an image.

Inputs:
* source image containing floor to be replaced
* new floor texture - square in dimensions. Optimally a couple tiles in width and height. 

Outputs:
* source image with the new floor texture replacing the original floor. 

Flooring visualiser is composed of the following components:
* Semantic Segmentation: recognising and masking the floor in the source.
* Texture Synthesis: using quilting techniques to generate more of the floor texture.
* Vanishing point estimation and Perspective Transformation: warping the new floor texture to match the perpective of the source image. 

TODO:
* shadows detection: apply shadows from the source image onto the new floor texture in the outputted image. 
