- test-severity-configs are basically how corrupted the image is
- gaussian-blur also has the area that should be blurred (kernel size) and how much blur (sigma)

- we create a deterministic validation/testing setup we use same seed so the corruption type is same for the image, the blurness,area mask coordinates and severity they remain consistent, we do this cuz for e.g if we have epoch 1 doing a tiny blur easy test but if in epoch 2 test got harder we wouldn't know if our model imporved/worsened or the test just changed

- we have a validation and testing manifesting generation process and save these manifests so that the corruptions are deterministic we can load these manifests every epoch to evaluate our model under same conditions
