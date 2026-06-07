import cv2
import numpy as np
import time
import json
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
import os
import datetime

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

            while True:
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
                
                    # Sort OCR results top-to-bottom
                    all_texts.sort(key=lambda t: t["box"][1])
                
                    sub_stats = []
                    stat_boxes = []
                
                    # Collect all text boxes in the stats region
                    log_debug(f"--- RAW STATS OCR DUMP ---")
                    for t in all_texts:
                        box = t.get("box", [0, 0, 0, 0])
                        x_box, y_box = box[0], box[1]
                        text_str = t.get("text", "").strip()
                        
                        if 30 <= x_box <= 560 and 280 <= y_box <= 450:
                            log_debug(f"  Box: {box} | Text: '{text_str}'")
                            if len(text_str) > 0 and text_str not in ["副属性值"]:
                                stat_boxes.append((x_box, y_box, text_str))
                    log_debug(f"--------------------------")
                
                    # Sort by Y (tolerance 15 pixels for same row), then X
                    stat_boxes.sort(key=lambda b: (b[1] // 15, b[0]))
                
                    current_key = None
                    import re
                
                    for x_stat, y_stat, text in stat_boxes:
                        if "尚未" in text:
                            sub_stats.append({"尚未炼成。": ""})
                            current_key = None
                        else:
                            text_clean = text.strip()
                            match = re.search(r"[\+\-]?(\d+(?:\.\d+)?%?)$", text_clean)
                            if match:
                                val_str = match.group(1)
                                # Try to parse as int or float if it's not a percentage, or just keep as string
                                # Keeping as string is safest to preserve format like '2.4%'
                                
                                # Extract the name part
                                name_part = text_clean[:match.start()].strip()
                                # Remove trailing colons, plus signs, and spaces
                                name_part = re.sub(r"[\+\s:]+$", "", name_part).strip()
                                
                                if name_part:
                                    sub_stats.append({name_part: val_str})
                                    current_key = None
                                else:
                                    if current_key:
                                        sub_stats.append({current_key: val_str})
                                        current_key = None
                            else:
                                # Text without numbers at the end -> probably a stat name
                                # Ignore common noise
                                if len(text) > 1:
                                    current_key = text
                
                    # Verify exactly 4 stats were extracted
                    assert len(sub_stats) == 4, f"Expected exactly 4 stats, but got {len(sub_stats)}: {sub_stats}"
                
                    # Find equipment name by checking if the OCR box falls within the target area
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
                        
                    # Create a unique key for this equipment based on its stats to avoid recording it multiple times
                    equip_key = f"{equip_name}_{'-'.join(str(s) for s in sub_stats)}"
                    if equip_key not in recorded_equipments:
                        recorded_equipments[equip_key] = True
                        processed_count += 1
                        log_debug(f"[{processed_count}] Name: {equip_name}")
                        log_debug(f"    Stats: {sub_stats}")
                        log_debug("-" * 30)

                log_debug("[check_equipment_reco] Finished visible rows. Swiping down...")
                # Swipe up on the grid area (scroll down the list)
                # from (900, 600) to (900, 200) over 400ms
                context.tasker.controller.post_swipe(900, 600, 900, 200, 400).wait()
                time.sleep(1.5) # Wait for scrolling animation to settle
            
            log_debug(f"[check_equipment_reco] Finished all equipment. Total processed: {processed_count}")
            
            # Once we are done recording, proceed to StopTask
            context.override_next(argv.node_name, ["StopTask"])
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Recorded all equipment")
            
        except Exception as e:
            import traceback
            log_debug(f"[check_equipment_reco] CRASHED WITH EXCEPTION: {e}")
            log_debug(traceback.format_exc())
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0), detail="Crashed")
