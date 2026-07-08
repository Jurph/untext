# LG Logo Aspect-Resize Plan

## Goal
Resize only images in `G:\Documents and Settings\Jurph\Old\My Pictures\Zero\has-LG-logo` whose aspect ratios match their peers very closely. Use a high-fidelity resampler (`Lanczos`) and leave outliers untouched.

## Grouping rule
Use a tight aspect-ratio tolerance:

- **Exact group**: images with aspect ratio exactly `0.7500`.
- **Near-3:4 group**: images within about **0.6% relative** of `0.7500`, but only when they form a coherent peer group.
- Do not force borderline outliers into a group just because they are close to one member.

## Observed groups

### Group A — exact 3:4
Target canvas: **`1632×2176`** (`AR 0.7500`)

Images:

- `watermark_bd2f5d41d29f4ffa9fa517e45d89a2f9_sophieraiin_95803845732.jpg` — `960×1280`
- `watermark_3fe46c31f38e494b905753b0b2281608_pinkchyu_88993365705.jpg` — `1080×1440`
- `watermark_a50ed015148f4679baf0f6aa83c5cd7e_pinkchyu_62840602274.jpg` — `1080×1440`
- `watermark_de6a6f42a5ad4296a0898e7c1f0e003d_pinkchyu_29590317862.jpg` — `1080×1440`
- `watermark_ec44d4295b974aea9288b97fc8b2e4b0_pinkchyu_43130730336.jpg` — `1080×1440`
- `watermark_6aaab38a38dc49e3814860c8784da4a9_sophieraiin_50609183824.jpg` — `1536×2048`
- `watermark_9cda24164dd342e1879a552b42d831db_fake_leximarvel_56231307659.png` — `1632×2176`

Resize all smaller images in this group to `1632×2176` with Lanczos.

### Group B — near-3:4
Target canvas: **`1640×2176`** (`AR 0.7537`)

Images:

- `watermark_a75ee82d59e640a2a6e8b11430b8ab4a_sophieraiin_41544850388.jpg` — `964×1280`
- `watermark_1de23103d797490cb37b5da2366b97f8_sophieraiin_46408840446.jpg` — `1080×1433`
- `watermark_43bb5b4c4577490db2fd1c80fc46c7d1_sophieraiin_3869991233.jpg` — `2880×3820`

Resize all three to `1640×2176` with Lanczos.

This gives the result the user asked for: two images grow, one shrinks.

### Leave untouched

- `watermark_ee598f5ef4f74d969e90e430607f26b1_sophieraiin_51913917218.png` — `728×1295` (`AR 0.5622`)
- `watermark_38af6b1f86bf4ce6b9efd323c35478a7_fake_leximarvel_59964542432.png` — `1552×2282` (`AR 0.6801`)
- `watermark_7858cf28fee64ee6ba037b795868edf3_sophieraiin_87965404153.jpg` — `970×1280` (`AR 0.7578`)

## Execution rules

1. Group by aspect ratio first.
2. Resize only groups that are internally tight.
3. Use Lanczos for all resizes.
4. Do not crop.
5. Do not touch outliers.

## Verification

Before any file change:

- Recompute image dimensions.
- Confirm the target group membership against the table above.
- Confirm the resize target for Group B is `1640×2176`, not `2880×3820`.

## Non-goals

- No content-aware cropping.
- No attempt to normalize unrelated aspect ratios.
- No changes to files outside `has-LG-logo`.
