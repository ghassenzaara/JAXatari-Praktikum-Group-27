# C51 on Pong: status

Group 27, Topic 27 (C51, IQN, Rainbow, IMPALA).

| Algo | Mode | Result (10M) | Reference |
|---|---|---|---|
| C51 | Pixel (CNN) | **Solves Pong, +20.8** | CleanRL C51 ~+20 |
| C51 | Object-centric (MLP) | **Solves Pong, +18.0** | no OC reference |

## Pixel

![pixel learning curve](c51_pixel_learning_curve.png)

## Object-centric

![object-centric learning curve](c51_object_centric_learning_curve.png)

**Finding:** object-centric observations needed normalization (raw pixel-coordinate features were saturating the MLP). Adding `NormalizeObservationWrapper` plus a 512x512 MLP fixed it. Same fix will apply to IQN, Rainbow and IMPALA.

Full metrics dashboard: `c51_object_centric_metrics.png`.

## Questions

1. Pixel matches the reference. Enough to call C51 pixel done, or do you want multi-seed variance bands?
2. OC uses `Normalize` + 512x512 (per your `ppo_oc`) instead of CleanRL's CartPole MLP. OK to keep?
3. Next: lock all four algorithms on Pong first, or take C51 straight to Seaquest?
