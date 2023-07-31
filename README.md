# Automated-Sign-Language-Feature-Extraction

This is the repository for pre-liminary experiments carried for the Automatic Sign Language Feature Extraction Tool (ASL-FET). The tool includes implementation of therotical morpho-phonological rules for

- Sign Boundary Detection
    - Major Location Constraint
    - Selected Finger Constraint

- Phonological Feature Detection:
    - Movement
        - Path Movement
        - Temporal Movement
    - Finger Selection
    - Location ( Major vs Minor Location )
    - Orientation (palm, back, tips, wrist, urnal, radial)

- Sign Detection (In-Progress)
    - Using Dynamic Time Warping utilizing both pose-estimation and phonological feature sets

## TODOs
- [ ] Normalize phonological feature series (for DTW)
- [ ] Async dtw inference
- [ ] Add / Merge Constraints for SBD
- [ ] Fix feature output format