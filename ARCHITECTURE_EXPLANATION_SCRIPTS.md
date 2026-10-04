# Simple Architecture Explanation Scripts

These scripts are written to be spoken aloud. Each one takes about one minute. The short flow line can be shown on screen while speaking.

## Task 1 - Universal Image Restoration

**Flow:** Damaged image -> encoder -> small feature map -> decoder -> restored image

**Say:**

> Task One uses one model to fix all three kinds of damage: noise, blur, and blocked areas. The model is an autoencoder with two main parts. First, the encoder makes the image smaller step by step. While doing this, it keeps the important information, such as shapes, colors, and edges. The smallest part is called the bottleneck. Next, the decoder grows this information back to the original image size and produces a clean image. The model is not told what type of damage it received. It must work that out by itself. So the easiest way to remember this architecture is: one model compresses the damaged image, understands its important features, and rebuilds a cleaner version.

## Task 2 - Hard-Routed Specialists

**Flow:** Image -> damage classifier -> one selected route -> restored image

**Say:**

> Task Two uses a team of specialist models. First, a small classifier looks at the image and chooses one of four labels: clean, noise, blur, or occlusion. Occlusion means that part of the image is covered. The classifier makes one hard decision, so only one route is selected. If the image is clean, it passes through unchanged. If it is damaged, it goes to the matching specialist. The noise specialist removes noise, the blur specialist sharpens blurred images, and the occlusion specialist fills missing areas. Each specialist is an encoder-decoder trained for only one job. In simple words, Task Two first asks, "What is wrong with this image?" and then sends it to the one expert designed to fix that problem.

## Task 3 - Soft Mixture of Experts

**Flow:** Image -> gating network -> four weights -> four branch outputs -> weighted blend

**Say:**

> Task Three keeps the same four branches from Task Two, but it does not choose only one. A small gating network gives a percentage to the clean route, the noise expert, the blur expert, and the occlusion expert. These four percentages always add up to one hundred percent. Every branch creates an output, and the system mixes those outputs using the four percentages. For example, the gate may say an image is mostly blurred but also slightly noisy. The final result can then use a large amount of the blur expert and a small amount of the noise expert. This is called soft routing. The easiest way to remember it is: Task Two picks one expert, but Task Three listens to all experts and blends their answers.

## Task 4 - Style-Conditioned Face-to-Sketch GAN

**Flow during training:** Face photo + style -> generator -> sketch -> discriminator feedback

**Flow in the application:** Face photo + style -> generator -> sketch

**Say:**

> Task Four changes a face photo into a sketch. During training, it uses two networks. The first network is the generator, which draws the sketch. It has a U-Net shape: it makes the photo smaller to understand the face, then makes it larger to create the sketch. Skip connections carry useful details, such as edges and face shape, from the first half to the second half. A learned style code tells the generator which of the three sketch styles to use. The second network is the discriminator. It checks small parts of the sketch and tells the generator whether they look real or fake. The two networks improve together. In the final application, only the generator is used, so the user provides a photo and style and receives a sketch.

## Very Short Backup Version

- **Task 1:** One autoencoder repairs every type of damage.
- **Task 2:** A classifier chooses one specialist to repair the image.
- **Task 3:** A gate blends the answers from all specialists.
- **Task 4:** A style-aware generator draws a sketch, while a discriminator helps train it to look real.
