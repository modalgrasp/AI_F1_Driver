# Yas Marina Track Acquisition Guide (Assetto Corsa)

## Goal
Acquire a high-quality Yas Marina Circuit mod for Assetto Corsa and install it safely for AI training.

## Trusted Sources
1. RaceDepartment (largest vetted AC mod community)
2. Assetto Corsa Club (community curation)
3. Reboot Team / known track authors' official pages
4. Sim racing Discord communities with verified mirrors

Avoid random file hosts with forced download managers, fake captcha installers, or executable installers.

## What to Download
Preferred archive formats:
1. `.zip` (best)
2. `.rar` (supported by this project tooling)

Expected mod structure (inside archive):
1. `content/tracks/<track_id>/...`
2. or `<track_id>/...` (where folder contains AC track files)

## High-Quality Mod Checklist
1. Includes `ui/ui_track.json` with metadata
2. Includes `models.ini` and at least one `.kn5` model
3. Includes `data/surfaces.ini`
4. Includes AI files in `ai/` (for example `fast_lane.ai`, `pit_lane.ai`)
5. Includes maps (`map.png`, `map.ini`, or equivalent)
6. Changelog and version history available
7. Author reputation and user ratings/comments are positive

## Version Selection Criteria
1. Prefer latest stable version with fixes for AI lines and track limits
2. Prefer versions tested with CSP versions close to your setup
3. Prefer versions with known layout IDs and sector metadata
4. Avoid "beta" unless no stable alternative exists

## Integrity and Safety Verification
1. Check archive extension matches content (`.zip` should open in 7-Zip)
2. If checksum is provided by author, verify:
   - `Get-FileHash .\yas_marina_mod.zip -Algorithm SHA256`
3. Compare approximate file size with source listing
4. Scan file with antivirus before extraction
5. Do not run `.exe` or `.bat` from track package

## Dependency Checks
Some track mods require:
1. CSP (Custom Shaders Patch)
2. Shared texture packs
3. Specific car packs for AI sessions

Read the mod description and install dependencies before validation.

## Installation Workflow (Recommended)
1. Download archive to `data/downloads/`
2. Validate archive and hash
3. Install with:
   - `python track_installer.py --archive data/downloads/<file>.zip --track-id yas_marina`
4. Validate with:
   - `python track_validator.py --track-id yas_marina`
5. Extract RL data with:
   - `python track_data_extractor.py --track-id yas_marina`

## Signs of a Bad Mod
1. Missing AI lane files
2. Missing `surfaces.ini`
3. Track length far from 5.281 km
4. Broken pit lane definitions
5. Frequent user reports of crashes or invisible walls

## Rollback Strategy
The installer in this project automatically creates backups before modifying track content. If installation fails, run rollback:
1. `python track_installer.py --rollback-latest`

## Security Notes
1. Use trusted source accounts with download history
2. Avoid repack archives with unknown branding
3. Keep mod archives immutable once validated (store hash in logs)
