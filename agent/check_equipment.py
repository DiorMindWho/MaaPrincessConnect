import cv2
import numpy as np
import time
import json
import re
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.custom_action import CustomAction
from maa.context import Context
import os
import datetime

global_recorded_equipments = []

LOG_FILE = r"D:\word\MaaFramework\MaaPrincessConnect\install\debug\equipment_log.txt"

def log_debug(msg: str):
    print(msg)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")
    except Exception as e:
        print(f"Failed to write to log: {e}")

def scan_sub_stats(context: Context, image: np.ndarray, base_x: int, base_y: int):
    stat_rois = [
        # Row 1
        {"type": "name", "roi": [0, 0, 200, 35]},
        {"type": "val",  "roi": [210, 0, 55, 35]},
        {"type": "name", "roi": [265, 0, 185, 35]},
        {"type": "val",  "roi": [450, 0, 80, 35]},
        # Row 2
        {"type": "name", "roi": [0, 38, 200, 35]},
        {"type": "val",  "roi": [210, 38, 55, 35]},
        {"type": "name", "roi": [265, 38, 185, 35]},
        {"type": "val",  "roi": [450, 38, 80, 35]},
    ]
    
    sub_stats = []
    current_key = None
    
    for idx, r in enumerate(stat_rois):
        dx, dy, w, h = r["roi"]
        x = base_x + dx
        y = base_y + dy
        crop_img = image[y:y+h, x:x+w]
        
        crop_override = {
            "OcrTask": {
                "recognition": "OCR",
                "expected": ""
            }
        }
        crop_reco = context.run_recognition("OcrTask", crop_img, pipeline_override=crop_override)
        
        text_str = ""
        if crop_reco and crop_reco.hit:
            text_str = "".join([t.get("text", "") for t in crop_reco.raw_detail.get("all", [])]).strip()
        
        if r["type"] == "name":
            if "尚未" in text_str:
                sub_stats.append({"尚未炼成。": ""})
                current_key = None
            else:
                name_clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text_str)
                if len(name_clean) > 1:
                    current_key = name_clean
                else:
                    current_key = None
        else:
            if current_key:
                val_clean = text_str.replace(" ", "")
                if val_clean:
                    sub_stats.append({current_key: val_clean})
                else:
                    sub_stats.append({current_key: "???"})
                current_key = None
                
    return sub_stats

@AgentServer.custom_recognition("check_equipment_reco")
class CheckEquipmentReco(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        try:
            log_debug("="*50)
            log_debug("[check_equipment_reco] Starting equipment grid iteration...")
            
            # Load templates
            img_dir = r"D:\word\MaaFramework\MaaPrincessConnect\assets\resource\image"
            ex_tpl = cv2.imread(rf"{img_dir}\ex.png")
            header_tpl = cv2.imread(rf"{img_dir}\equipmentheader.png")
            
            if ex_tpl is None or header_tpl is None:
                log_debug("[check_equipment_reco] Missing template images.")
                return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Templates missing")

            last_grid_img = None
            processed_count = 0
        
            # Dictionary to store recorded items to avoid printing duplicates across scrolls
            recorded_equipments = {}
            global global_recorded_equipments
            global_recorded_equipments.clear()
        
            # On the first page, we start from the 1st equipment. 
            # On subsequent pages, we skip the first 5 because they were the bottom row of the previous page.
            start_idx = 0
            
            while True:
                if context.tasker.stopping:
                    log_debug("[check_equipment_reco] Task stopped by user.")
                    return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Stopped")
                    
                # 1. Take fresh screenshot
                image_future = context.tasker.controller.post_screencap().wait()
                image = image_future.get()
                if image is None:
                    log_debug("[check_equipment_reco] Failed to get screenshot.")
                    break
                
                # 2. Find header to define grid Y-boundary
                res_h = cv2.matchTemplate(image, header_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val_h, _, max_loc_h = cv2.minMaxLoc(res_h)
                header_y = max_loc_h[1] + header_tpl.shape[0] if max_val_h >= 0.7 else 0
            
                # 3. Find EX icons
                res_ex = cv2.matchTemplate(image, ex_tpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res_ex >= 0.8)
                ex_pts = list(zip(*loc[::-1]))
            
                # Filter overlapping and left-panel EX icons
                filtered_ex = []
                for pt in ex_pts:
                    # Ignore left panel details EX icon
                    if pt[0] < 550:
                        continue
                    # Ignore anything above header
                    if pt[1] < header_y:
                        continue
                    
                    if not any(abs(pt[0]-fp[0]) < 20 and abs(pt[1]-fp[1]) < 20 for fp in filtered_ex):
                        filtered_ex.append(pt)
                    
                # Sort by Y (rows), then X (columns)
                # Group by row: if Y difference is < 40, they are in the same row
                filtered_ex.sort(key=lambda p: (p[1] // 40, p[0]))
            
                if not filtered_ex:
                    log_debug("[check_equipment_reco] No equipment found in grid. Ending.")
                    break
                
                # 4. Check if grid has stopped moving (reached bottom)
                # We crop the entire grid area on the right
                grid_crop = image[header_y:720, 550:1280]
                if last_grid_img is not None:
                    # Calculate absolute difference between previous grid and current grid
                    diff = cv2.absdiff(grid_crop, last_grid_img)
                    if np.mean(diff) < 2.0:  # Very little change -> we hit the bottom
                        log_debug("[check_equipment_reco] Grid hasn't moved. Reached bottom of the equipment list.")
                        break
                last_grid_img = grid_crop.copy()
            
                # 5. Process each equipment in the currently visible grid
                for idx, (x, y) in enumerate(filtered_ex):
                    if idx < start_idx:
                        continue
                        
                    log_debug(f"[check_equipment_reco] Processing equipment {idx+1}/{len(filtered_ex)} at ({x}, {y})")
                
                    # Click the equipment (offset by half the EX icon size to hit the center of the equipment)
                    # MUST cast to int because x and y are numpy.int64, which crashes C++ bindings
                    click_x = int(x + ex_tpl.shape[1] // 2 + 20)
                    click_y = int(y + ex_tpl.shape[0] // 2 + 20)
                
                    context.tasker.controller.post_click(click_x, click_y).wait()
                    time.sleep(0.8) # Wait for left panel to render new details
                
                    # Take screenshot to run OCR on the left panel
                    panel_img = context.tasker.controller.post_screencap().wait().get()
                    if panel_img is None:
                        continue
                    
                    # Run OCR specifically bounded to the left panel (x < 550)
                    # We use the existing "OcrTask" node defined in my_task.json to avoid "node not found" errors
                    override = {
                        "OcrTask": {
                            "recognition": "OCR",
                            "expected": "",
                            "roi": [0, 0, 550, 720]
                        }
                    }
                    reco_detail = context.run_recognition("OcrTask", panel_img, pipeline_override=override)
                
                    if not reco_detail or not reco_detail.hit:
                        continue
                    
                    # The OCR results are already parsed into a dictionary by the Python API
                    try:
                        ocr_data = reco_detail.raw_detail
                        all_texts = ocr_data.get("all", [])
                    except Exception as e:
                        log_debug(f"Failed to extract OCR data: {e}")
                        continue
                
                    # We keep the full-panel OCR just for the equipment name since it works perfectly
                    equip_name = "Unknown Equipment"
                    # User specified area: x: 140 to 400, y: 100 to 140
                    for t in all_texts:
                        box = t.get("box", [0, 0, 0, 0])
                        x_box, y_box = box[0], box[1]
                        text_str = t.get("text", "").strip()
                        
                        # We add a small tolerance margin to the coordinates to be safe
                        if 120 <= x_box <= 420 and 90 <= y_box <= 150:
                            if len(text_str) > 1 and text_str not in ["取消", "返回"]:
                                equip_name = text_str
                                break
                    
                    # Targeted ROI Scanning for Sub-Stats
                    sub_stats = scan_sub_stats(context, panel_img, 30, 310)
                    
                    # Verify exactly 4 stats were extracted
                    assert len(sub_stats) == 4, f"Expected exactly 4 stats, but got {len(sub_stats)}: {sub_stats}"
                        
                    # Create a unique key for this equipment based on its stats to avoid recording it multiple times
                    equip_key = f"{equip_name}_{'-'.join(str(s) for s in sub_stats)}"
                    if equip_key not in recorded_equipments:
                        recorded_equipments[equip_key] = True
                        global_recorded_equipments.append({
                            "name": equip_name,
                            "stats": sub_stats
                        })
                        processed_count += 1
                        log_debug(f"[{processed_count}] Name: {equip_name}")
                        log_debug(f"    Stats: {sub_stats}")
                        log_debug("-" * 30)

                log_debug("[check_equipment_reco] Finished visible rows. Swiping down...")
                # Swipe exactly one row up by dragging the 10th equipment straight up to the 5th equipment's Y level
                if len(filtered_ex) >= 10:
                    x_10, y_10 = filtered_ex[9]
                    _, y_5 = filtered_ex[4]
                    context.tasker.controller.post_swipe(int(x_10), int(y_10), int(x_10), int(y_5), 400).wait()
                else:
                    context.tasker.controller.post_swipe(900, 600, 900, 200, 400).wait()
                    
                time.sleep(1.5) # Wait for scrolling animation to settle
                
                # The old bottom row (equipments 6-10) is now the new top row (equipments 1-5).
                # We already scanned them, so start scanning from the 6th equipment (index 5) on the next page!
                start_idx = 5
            
            log_debug(f"[check_equipment_reco] Finished all equipment. Total processed: {processed_count}")
            
            # Once we are done recording, proceed to EnhanceEquipment
            context.override_next(argv.node_name, ["EnhanceEquipment"])
            return CustomRecognition.AnalyzeResult(box=(0,0,1,1), detail="Recorded all equipment")
            
        except Exception as e:
            import traceback
            log_debug(f"[check_equipment_reco] CRASHED WITH EXCEPTION: {e}")
            log_debug(traceback.format_exc())
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Crashed")

@AgentServer.custom_action("enhance_equipment_action")
class EnhanceEquipmentAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            log_debug("="*50)
            log_debug("[enhance_equipment_action] Analyzing equipment to enhance...")
            
            global global_recorded_equipments
            
            lowest_val = float('inf')
            lowest_equip = None
            
            for equip in global_recorded_equipments:
                stats = equip.get("stats", [])
                
                # sum up 物理防御贯穿 or 魔法防御贯穿
                total_penetration = 0
                
                for stat_dict in stats:
                    for key, val in stat_dict.items():
                        if key in ["物理防御贯穿", "魔法防御贯穿"]:
                            try:
                                # Extract digits in case of "???" or other symbols
                                num_str = re.sub(r"\D", "", str(val))
                                if num_str:
                                    total_penetration += int(num_str)
                            except ValueError:
                                pass
                
                if total_penetration < lowest_val:
                    lowest_val = total_penetration
                    lowest_equip = equip
            
            if lowest_equip:
                log_debug(f"[enhance_equipment_action] Lowest penetration equipment to enhance:")
                log_debug(f"    Name: {lowest_equip['name']}")
                log_debug(f"    Stats: {lowest_equip['stats']}")
                log_debug(f"    Total Penetration: {lowest_val}")
            else:
                log_debug("[enhance_equipment_action] No equipment found.")
                return True
                
            # --- NEW WORKFLOW ---
            # 1. Scroll back to top
            log_debug("[enhance_equipment_action] Scrolling back to top...")
            for _ in range(3):
                context.tasker.controller.post_swipe(900, 200, 900, 600, 300).wait()
                time.sleep(0.5)
            
            # Wait for list to settle
            time.sleep(2)
            
            # 2. Iterate and select the target equipment
            img_dir = r"D:\word\MaaFramework\MaaPrincessConnect\assets\resource\image"
            ex_tpl = cv2.imread(rf"{img_dir}\ex.png")
            header_tpl = cv2.imread(rf"{img_dir}\equipmentheader.png")
            
            target_found = False
            last_grid_img = None
            
            while not target_found:
                if context.tasker.stopping:
                    log_debug("[enhance_equipment_action] Task stopped by user.")
                    return False
                    
                image_future = context.tasker.controller.post_screencap().wait()
                image = image_future.get()
                if image is None:
                    break
                    
                res_h = cv2.matchTemplate(image, header_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val_h, _, max_loc_h = cv2.minMaxLoc(res_h)
                header_y = max_loc_h[1] + header_tpl.shape[0] if max_val_h >= 0.7 else 0
                
                res_ex = cv2.matchTemplate(image, ex_tpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res_ex >= 0.8)
                ex_pts = list(zip(*loc[::-1]))
                
                filtered_ex = []
                for pt in ex_pts:
                    if pt[0] < 550: continue
                    if pt[1] < header_y: continue
                    if not any(abs(pt[0]-fp[0]) < 20 and abs(pt[1]-fp[1]) < 20 for fp in filtered_ex):
                        filtered_ex.append(pt)
                        
                filtered_ex.sort(key=lambda p: (p[1] // 40, p[0]))
                
                if not filtered_ex:
                    break
                    
                grid_crop = image[header_y:720, 550:1280]
                if last_grid_img is not None:
                    diff = cv2.absdiff(grid_crop, last_grid_img)
                    if np.mean(diff) < 2.0:
                        break
                last_grid_img = grid_crop.copy()
                
                for idx, (x, y) in enumerate(filtered_ex):
                    click_x = int(x + ex_tpl.shape[1] // 2 + 20)
                    click_y = int(y + ex_tpl.shape[0] // 2 + 20)
                    
                    context.tasker.controller.post_click(click_x, click_y).wait()
                    time.sleep(0.8)
                    
                    panel_img = context.tasker.controller.post_screencap().wait().get()
                    if panel_img is None: continue
                    
                    override = { "OcrTask": { "recognition": "OCR", "expected": "", "roi": [120, 90, 300, 60] } }
                    reco_detail = context.run_recognition("OcrTask", panel_img, pipeline_override=override)
                    
                    equip_name = "Unknown"
                    if reco_detail and reco_detail.hit:
                        texts = [t.get("text", "").strip() for t in reco_detail.raw_detail.get("all", [])]
                        for text_str in texts:
                            if len(text_str) > 1 and text_str not in ["取消", "返回"]:
                                equip_name = text_str
                                break
                    
                    if equip_name == lowest_equip["name"]:
                        current_stats = scan_sub_stats(context, panel_img, 30, 310)
                        if str(current_stats) == str(lowest_equip["stats"]):
                            log_debug(f"[enhance_equipment_action] Target equipment found and selected!")
                            target_found = True
                            break
                            
                if target_found:
                    break
                    
                if len(filtered_ex) >= 10:
                    x_10, y_10 = filtered_ex[9]
                    _, y_5 = filtered_ex[4]
                    context.tasker.controller.post_swipe(int(x_10), int(y_10), int(x_10), int(y_5), 400).wait()
                else:
                    context.tasker.controller.post_swipe(900, 600, 900, 200, 400).wait()
                time.sleep(1.5)
                
            if not target_found:
                log_debug("[enhance_equipment_action] Could not find the target equipment in the list.")
                return False
                
            return True
            
        except Exception as e:
            import traceback
            log_debug(f"[enhance_equipment_action] CRASHED WITH EXCEPTION: {e}")
            log_debug(traceback.format_exc())
            return False

@AgentServer.custom_action("refine_equipment_action")
class RefineEquipmentAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            log_debug("="*50)
            log_debug("[refine_equipment_action] Starting refining loop...")
            img_dir = r"D:\word\MaaFramework\MaaPrincessConnect\assets\resource\image"
            
            # --- The Refining Loop ---
            skip_buxianshi = False
            
            while True:
                if context.tasker.stopping:
                    log_debug("[enhance_equipment_action] Task stopped by user during refine.")
                    return False
                    
                log_debug("[enhance_equipment_action] Looking for liancheng.png")
                liancheng_tpl = cv2.imread(rf"{img_dir}\liancheng.png")
                clicked_liancheng = False
                for _ in range(5):
                    image = context.tasker.controller.post_screencap().wait().get()
                    if liancheng_tpl is not None:
                        res = cv2.matchTemplate(image, liancheng_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val >= 0.8:
                            cx = max_loc[0] + liancheng_tpl.shape[1] // 2
                            cy = max_loc[1] + liancheng_tpl.shape[0] // 2
                            context.tasker.controller.post_click(cx, cy).wait()
                            clicked_liancheng = True
                            break
                    time.sleep(1)
                    
                if not clicked_liancheng:
                    log_debug("[enhance_equipment_action] Could not find liancheng.png")
                    break
                
                # Check for confirm_first.png
                log_debug("[enhance_equipment_action] Looking for confirm_first.png")
                confirm_first_tpl = cv2.imread(rf"{img_dir}\confirm_first.png")
                for _ in range(5):
                    image = context.tasker.controller.post_screencap().wait().get()
                    if confirm_first_tpl is not None:
                        res = cv2.matchTemplate(image, confirm_first_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val >= 0.8:
                            cx = max_loc[0] + confirm_first_tpl.shape[1] // 2
                            cy = max_loc[1] + confirm_first_tpl.shape[0] // 2
                            context.tasker.controller.post_click(cx, cy).wait()
                            break
                    time.sleep(1)
                
                log_debug("[enhance_equipment_action] Waiting for result indication...")
                resultind_tpl = cv2.imread(rf"{img_dir}\resultindication.png")
                found_result = False
                for _ in range(15):
                    time.sleep(1)
                    image = context.tasker.controller.post_screencap().wait().get()
                    if resultind_tpl is not None:
                        res = cv2.matchTemplate(image, resultind_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        if max_val >= 0.8:
                            found_result = True
                            break
                        
                if not found_result:
                    log_debug("[enhance_equipment_action] resultindication.png not found")
                    break
                
                time.sleep(1)
                image = context.tasker.controller.post_screencap().wait().get()
                current_stats = scan_sub_stats(context, image, 65, 517)
                incoming_stats = scan_sub_stats(context, image, 711, 517)
                
                def get_pen_stats(stats_list):
                    tot = 0
                    vals = []
                    for stat_dict in stats_list:
                        for key, val in stat_dict.items():
                            if key in ["物理防御贯穿", "魔法防御贯穿"]:
                                try:
                                    num_str = re.sub(r"\D", "", str(val))
                                    if num_str: 
                                        v = int(num_str)
                                        tot += v
                                        vals.append(v)
                                except ValueError:
                                    pass
                    vals.sort(reverse=True)
                    return tot, vals
                    
                cur_pen, cur_vals = get_pen_stats(current_stats)
                inc_pen, inc_vals = get_pen_stats(incoming_stats)
                
                log_debug(f"[enhance_equipment_action] Current Pen: {cur_pen} (Vals: {cur_vals}), Incoming Pen: {inc_pen} (Vals: {inc_vals})")
                
                keep_incoming = False
                if inc_pen > cur_pen:
                    keep_incoming = True
                elif inc_pen == cur_pen and inc_vals > cur_vals:
                    keep_incoming = True
                
                if keep_incoming:
                    log_debug("[enhance_equipment_action] Incoming is better. Clicking confirm.")
                    btn_tpl = cv2.imread(rf"{img_dir}\confirm.png")
                else:
                    log_debug("[enhance_equipment_action] Incoming is not better. Clicking revert.")
                    btn_tpl = cv2.imread(rf"{img_dir}\revert.png")
                    
                if btn_tpl is not None:
                    res = cv2.matchTemplate(image, btn_tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.8:
                        cx = max_loc[0] + btn_tpl.shape[1] // 2
                        cy = max_loc[1] + btn_tpl.shape[0] // 2
                        context.tasker.controller.post_click(cx, cy).wait()
                    else:
                        log_debug("[enhance_equipment_action] Confirm/Revert button not found!")
                else:
                    log_debug("[enhance_equipment_action] Confirm/Revert template image missing!")
                    
                time.sleep(1.5)
                
                if not skip_buxianshi:
                    image = context.tasker.controller.post_screencap().wait().get()
                    buxianshi_tpl = cv2.imread(rf"{img_dir}\buxianshi.png")
                    checkbox_tpl = cv2.imread(rf"{img_dir}\checkbox.png")
                    
                    override = { "OcrTask": { "recognition": "OCR", "expected": "不显示今后的消息" } }
                    reco_detail = context.run_recognition("OcrTask", image, pipeline_override=override)
                    
                    max_val_bx = 0
                    if buxianshi_tpl is not None:
                        res_bx = cv2.matchTemplate(image, buxianshi_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val_bx, _, _ = cv2.minMaxLoc(res_bx)
                    
                    if (reco_detail and reco_detail.hit) or max_val_bx >= 0.8:
                        log_debug("[enhance_equipment_action] Handling 'Do not show again' dialog.")
                        if checkbox_tpl is not None:
                            res_cb = cv2.matchTemplate(image, checkbox_tpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val_cb, _, max_loc_cb = cv2.minMaxLoc(res_cb)
                            if max_val_cb >= 0.8:
                                context.tasker.controller.post_click(max_loc_cb[0] + 10, max_loc_cb[1] + 10).wait()
                                time.sleep(0.5)
                            
                        override_conf = { "OcrTask": { "recognition": "OCR", "expected": "确认" } }
                        reco_conf = context.run_recognition("OcrTask", image, pipeline_override=override_conf)
                        if reco_conf and reco_conf.hit:
                            box = reco_conf.box
                            context.tasker.controller.post_click(box[0] + box[2]//2, box[1] + box[3]//2).wait()
                            time.sleep(1)
                    
                    # We only need to check it once. Subsequent loops don't need this check.
                    skip_buxianshi = True
                
                time.sleep(1)
                image = context.tasker.controller.post_screencap().wait().get()
                panel_stats = scan_sub_stats(context, image, 30, 310)
                
                lockoff_tpl = cv2.imread(rf"{img_dir}\lockoff.png")
                
                maxed_pen_count = 0
                total_pen_count = 0
                
                stat_rois_vals = [
                    [240, 310, 55, 35], [480, 310, 80, 35],
                    [240, 348, 55, 35], [480, 348, 80, 35]
                ]
                
                for idx, stat_dict in enumerate(panel_stats):
                    for key, val in stat_dict.items():
                        if key in ["物理防御贯穿", "魔法防御贯穿"]:
                            total_pen_count += 1
                            try:
                                num_str = re.sub(r"\D", "", str(val))
                                if num_str:
                                    val_int = int(num_str)
                                    if val_int == 3:
                                        maxed_pen_count += 1
                                        if lockoff_tpl is not None and idx < len(stat_rois_vals):
                                            vx, vy, vw, vh = stat_rois_vals[idx]
                                            
                                            # Expand the search area to catch the lock icon correctly
                                            # The lock icon is around 80px to the left of the value (e.g. x=160 when value is x=240)
                                            search_x = max(0, vx - 100)
                                            search_y = max(0, vy - 20)
                                            search_w = 120
                                            search_h = vh + 40
                                            
                                            crop_area = image[search_y:search_y+search_h, search_x:search_x+search_w]
                                            res_lo = cv2.matchTemplate(crop_area, lockoff_tpl, cv2.TM_CCOEFF_NORMED)
                                            _, max_val_lo, _, max_loc_lo = cv2.minMaxLoc(res_lo)
                                            
                                            if max_val_lo >= 0.8:
                                                lx = search_x + max_loc_lo[0] + lockoff_tpl.shape[1] // 2
                                                ly = search_y + max_loc_lo[1] + lockoff_tpl.shape[0] // 2
                                                context.tasker.controller.post_click(lx, ly).wait()
                                                time.sleep(0.5)
                            except ValueError:
                                pass
                                
                if maxed_pen_count == 4 or (maxed_pen_count == 3 and total_pen_count == 4):
                    log_debug(f"[enhance_equipment_action] Stop condition met! Maxed: {maxed_pen_count}, Total Pen Stats: {total_pen_count}. Refine complete.")
                    break
                
            return True
            
        except Exception as e:
            import traceback
            log_debug(f"[enhance_equipment_action] CRASHED WITH EXCEPTION: {e}")
            log_debug(traceback.format_exc())
            return False
