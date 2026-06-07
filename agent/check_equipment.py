import cv2
import numpy as np
import time
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

@AgentServer.custom_recognition("check_equipment_reco")
class CheckEquipmentReco(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        image = argv.image
        
        # Load templates
        img_dir = r"D:\word\MaaFramework\MaaPrincessConnect\assets\resource\image"
        ex_tpl = cv2.imread(rf"{img_dir}\ex.png")
        magic_tpl = cv2.imread(rf"{img_dir}\fullmagic.png")
        physic_tpl = cv2.imread(rf"{img_dir}\fullphysic.png")
        header_tpl = cv2.imread(rf"{img_dir}\equipmentheader.png")
        
        if any(t is None for t in [ex_tpl, magic_tpl, physic_tpl, header_tpl]):
            print("check_equipment_reco: Missing template images.")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Templates missing")

        # Create masks to ignore the lock icons in the templates
        # We assume the template has 2 columns, lock icons are roughly around 30%-45% and 80%-95% of width.
        h, w = magic_tpl.shape[:2]
        mask = np.ones_like(magic_tpl) * 255
        mask[:, int(w*0.30):int(w*0.48)] = 0
        mask[:, int(w*0.80):int(w*0.98)] = 0

        # 1. Find equipment header
        res_h = cv2.matchTemplate(image, header_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val_h, _, max_loc_h = cv2.minMaxLoc(res_h)
        
        header_y = max_loc_h[1] + header_tpl.shape[0] if max_val_h >= 0.7 else 0
        
        # Crop below header to search for EX icons
        roi_img = image[header_y:, :]
        
        # 2. Find EX icons
        res_ex = cv2.matchTemplate(roi_img, ex_tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res_ex >= 0.8)
        ex_pts = list(zip(*loc[::-1]))
        
        # Filter overlapping EX icons
        filtered_ex = []
        for pt in ex_pts:
            if not any(abs(pt[0]-fp[0]) < 20 and abs(pt[1]-fp[1]) < 20 for fp in filtered_ex):
                filtered_ex.append(pt)
                
        # 3. Check each equipment block
        # We'll search around the EX icon for the maxed stats
        box_w, box_h = int(w * 1.2), int(h * 1.5)
        
        all_maxed = True
        selected_click = None

        for (x, y) in filtered_ex:
            # The stats block usually appears below or next to the EX icon
            # Let's crop a box large enough to contain the stats block
            search_box = roi_img[max(0, y):y+box_h, max(0, x):x+box_w]
            
            if search_box.shape[0] < h or search_box.shape[1] < w:
                continue
                
            # Use TM_SQDIFF_NORMED with mask: perfect match is 0.0
            res_m = cv2.matchTemplate(search_box, magic_tpl, cv2.TM_SQDIFF_NORMED, mask=mask)
            _, min_m, _, _ = cv2.minMaxLoc(res_m)
            
            res_p = cv2.matchTemplate(search_box, physic_tpl, cv2.TM_SQDIFF_NORMED, mask=mask)
            _, min_p, _, _ = cv2.minMaxLoc(res_p)
            
            # If min is small enough, it means the stats perfectly match the maxed template
            is_magic_max = min_m < 0.15
            is_physic_max = min_p < 0.15
            
            print(f"[check_equipment_reco] EX icon at ({x}, {y}) -> Magic diff: {min_m:.4f}, Physic diff: {min_p:.4f}")
            
            if not (is_magic_max or is_physic_max):
                print(f"[check_equipment_reco] FOUND UNMAXED EQUIPMENT at ({x}, {y})!")
                # This equipment is NOT maxed out! We should select it.
                # Click the center of this equipment box
                click_x = x + ex_tpl.shape[1] // 2
                click_y = header_y + y + ex_tpl.shape[0] // 2
                selected_click = (click_x, click_y)
                all_maxed = False
                break
        
        if selected_click:
            print(f"[check_equipment_reco] Stopping task and clicking unmaxed equipment at {selected_click}")
            # We found one!
            context.override_next(argv.node_name, ["StopTask"])
            return CustomRecognition.AnalyzeResult(
                box=(selected_click[0], selected_click[1], 10, 10),
                detail="Selected unmaxed equipment"
            )
        
        print("[check_equipment_reco] All visible equipments are maxed out. Swiping down...")
        # If no EX found or all are maxed, we need to swipe down and loop
        # We override the next node to ourselves to create a loop
        context.override_next(argv.node_name, ["SelectEquipment"])
        
        # Swipe up (scroll down the list)
        swipe_job = context.tasker.controller.post_swipe(640, 500, 640, 200, 400)
        swipe_job.wait()
        
        # Add a short delay for the animation to settle
        time.sleep(1.5)
        
        # Return an empty box to let the pipeline proceed to the overridden next node
        return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Swiped down, continuing search")
