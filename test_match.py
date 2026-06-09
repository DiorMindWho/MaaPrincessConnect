import cv2
import numpy as np

def test_match():
    # Load images
    screenshot_path = 'assets/resource/image/screenshot.png'
    template_path = 'assets/resource/image/lockoff.png'
    
    img = cv2.imread(screenshot_path)
    tmpl = cv2.imread(template_path)
    
    if img is None:
        print(f"Error: Could not load screenshot from {screenshot_path}")
        return
    if tmpl is None:
        print(f"Error: Could not load template from {template_path}")
        return
        
    print(f"Screenshot size: {img.shape}")
    print(f"Template size: {tmpl.shape}")
    
    # Perform match
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    
    # Get max score and location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"Max match score: {max_val:.4f} (Requires >= 0.80 for MaaFramework)")
    print(f"Best match location (x, y): {max_loc}")
    
    # Draw rectangle around the best match
    h, w = tmpl.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
    
    # Save visualization
    out_path = 'install/debug/match_result_visualization.png'
    cv2.imwrite(out_path, img)
    print(f"Saved visualization to {out_path} so you can see exactly where it matched!")

if __name__ == "__main__":
    test_match()
