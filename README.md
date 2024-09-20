# CycleGAN-Illumination

## Abstract
Synthesis of realistic virtual environments requires careful rendering of light and shadows, a task often bottle-necked by the high computational cost of global illumination (GI) techniques. This paper introduces a new GI approach that improves computational efficiency without a significant reduction in image quality. The proposed system transforms initial direct-illumination renderings into globally illuminated representations by incorporating a Cycle-Consistent Adversarial Network (CycleGAN). Our CycleGAN-based approach has demonstrated superior performance over the Pix2Pix model according to the LPIPS metric, which emphasizes perceptual similarity. To facilitate such comparisons, we have created a novel dataset (to be shared with the research community) that provides in-game images that were obtained with and without GI rendering. This work aims to advance real-time GI estimation without the need for costly, specialized computational hardware. The paper can be found here (soon to be linked).

## Note
1. Only intended to be used in a Linux Environment.
2. For GPU support, ensure you install PyTorch with the correct CUDA version. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for detailed instructions.


## Setup
1. Clone the repository.
   ```console
    git clone https://github.com/junhofive/CycleGAN-Illumination.git
    ```

2. Run the Setup Script.
   ```console
    sh setup.sh
    ```

3. Run the Main Program.
