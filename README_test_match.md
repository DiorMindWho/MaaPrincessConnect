# testing matches

The `test_match.py` script is a handy debug tool to verify if a template image (e.g. `gearselect.png`) can successfully be matched inside a full screenshot using OpenCV's `matchTemplate` function, which is the exact same matching engine used by MaaFramework.

## How to use

1. Open `test_match.py` and modify the paths to your screenshot and template image:
   ```python
   screenshot_path = 'install/debug/on_error/...png'
   template_path = 'assets/resource/image/...png'
   ```

2. Run the script from the root directory using the virtual environment:
   ```powershell
   .venv\Scripts\python.exe test_match.py
   ```

3. The script will print the maximum match score. If the score is `< 0.80`, MaaFramework will fail to find it.

4. The script also saves an output image to `install/debug/match_result_visualization.png` with a red box drawn exactly where it thought the best match was, so you can visually verify if it matched the correct area.
