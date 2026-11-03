"""English <-> Thai switch for the app (button in the top bar).

The pages are written in English; when Thai is picked, a small script in the
page swaps every text it knows for the Thai one, also text that the page adds
later (job progress, the video panel, answers). Unknown text stays English.
To add a translation: put the exact English text in EXACT, or a pattern in
PATTERNS (for text with numbers or names in it).
"""
import json

# behavior and state names (also used inside longer texts)
WORDS = {
    "Inactive": "ไม่เคลื่อนไหว",
    "Active": "เคลื่อนไหว",
    "Not visible": "มองไม่เห็น",
    "Feeding": "กินอาหาร",
    "Drinking": "ดื่มน้ำ",
    "Territorial marking": "ทำเครื่องหมายอาณาเขต",
    "Grooming": "ทำความสะอาดตัว",
    "Urination": "ปัสสาวะ",
    "Defecation": "อุจจาระ",
    "Urine": "ปัสสาวะ",
    "Feces": "อุจจาระ",
    "Locomotion / Walking": "เดิน",
    "Walking": "เดิน",
    "Active (other)": "เคลื่อนไหวอื่นๆ",
    "Lying down": "นอนราบ",
    "Curling up": "นอนขดตัว",
    "Stretched out": "นอนเหยียดตัว",
    "Sleep in a box": "นอนในกล่อง",
    "Poor image": "ภาพไม่ชัด",
    "Mating Behavior": "พฤติกรรมผสมพันธุ์",
    "Short stop (check)": "หยุดนิ่งช่วงสั้น (รอตรวจ)",
    "Passing between rooms": "เดินผ่านระหว่างห้อง",
    "On the sleeping box top (check)": "บนหลังคากล่องนอน (รอตรวจ)",
    "Possible urination / defecation (check)": "อาจปัสสาวะ / อุจจาระ (รอตรวจ)",
    "Possible marking (check)": "อาจทำเครื่องหมาย (รอตรวจ)",
    "Possible grooming (check)": "อาจทำความสะอาดตัว (รอตรวจ)",
}

EXACT = {
    **WORDS,
    # top bar and tabs
    "Chiang Mai Zoo": "สวนสัตว์เชียงใหม่",
    "+ Process new footage": "+ ประมวลผลวิดีโอใหม่",
    "Observation day": "วันที่บันทึก",
    "Overview": "ภาพรวม",
    "Observation sheet": "ตารางบันทึกพฤติกรรม",
    "Charts": "กราฟ",
    "Events & video": "เหตุการณ์และวิดีโอ",
    "Behavior check": "ตรวจพฤติกรรม",
    "Spot check": "สุ่มตรวจความแม่นยำ",
    "Files": "ไฟล์",
    "Esc · close ✕": "Esc · ปิด ✕",
    # home
    "Observation days": "วันที่บันทึกทั้งหมด",
    "Remove this day": "ลบวันนี้",
    "Remove": "ลบ",
    "all checked": "ตรวจครบแล้ว",
    "Recent jobs": "งานล่าสุด",
    "Started": "เริ่ม",
    "Job": "งาน",
    "Result": "ผล",
    "Took": "ใช้เวลา",
    "done": "เสร็จ",
    "failed": "ผิดพลาด",
    "cancelled": "หยุดแล้ว",
    "running": "กำลังทำงาน",
    "see progress": "ดูความคืบหน้า",
    "progress": "ความคืบหน้า",
    "is running": "กำลังทำงาน",
    # new footage
    "Process new footage": "ประมวลผลวิดีโอใหม่",
    "CivetWatch reads the 4 cameras' recordings of one day, finds the civet in every frame and fills the observation sheet. "
    "The videos stay where they are; a small 2-frames-per-second copy is made in":
        "CivetWatch อ่านวิดีโอจากกล้องทั้ง 4 ตัวของหนึ่งวัน หาชะมดในทุกภาพ แล้วกรอกตารางบันทึกพฤติกรรมให้ "
        "วิดีโอต้นฉบับอยู่ที่เดิม โปรแกรมจะทำสำเนาขนาดเล็ก (2 ภาพต่อวินาที) ไว้ที่",
    ". Zones and settings are copied from the last day (the cameras do not move).":
        " โซนและการตั้งค่าคัดลอกจากวันล่าสุด (กล้องไม่ได้ย้ายตำแหน่ง)",
    "1 · Day": "1 · วันที่",
    "filled in from the file names when you pick a folder": "กรอกให้อัตโนมัติจากชื่อไฟล์เมื่อเลือกโฟลเดอร์",
    "2 · Where are the videos?": "2 · วิดีโออยู่ที่ไหน?",
    "A folder on this computer (recommended)": "โฟลเดอร์ในเครื่องนี้ (แนะนำ)",
    "Upload video files": "อัปโหลดไฟล์วิดีโอ",
    "One folder with a sub-folder per camera (\"camera 1\" ... \"camera 4\"), or type each camera's folder.":
        "โฟลเดอร์เดียวที่มีโฟลเดอร์ย่อยของแต่ละกล้อง (\"camera 1\" ... \"camera 4\") หรือพิมพ์โฟลเดอร์ของแต่ละกล้อง",
    "or one folder per camera": "หรือกำหนดโฟลเดอร์ของแต่ละกล้อง",
    "The files are copied into": "ไฟล์จะถูกคัดลอกไปที่",
    "(needs free space for the full recordings). File names must contain the date and time, like":
        "(ต้องมีพื้นที่ว่างพอสำหรับวิดีโอทั้งหมด) ชื่อไฟล์ต้องมีวันที่และเวลา เช่น",
    "Check footage and estimate time": "ตรวจวิดีโอและประมาณเวลา",
    "Checking the videos ...": "กำลังตรวจวิดีโอ ...",
    "Found": "พบวิดีโอ",
    "Camera": "กล้อง",
    "Folder": "โฟลเดอร์",
    "Videos": "วิดีโอ",
    "Hours": "ชั่วโมง",
    "Estimated time": "เวลาโดยประมาณ",
    "in total, in the background": "รวมทั้งหมด (ทำงานเบื้องหลัง)",
    "prepare videos": "เตรียมวิดีโอ",
    "find the civet + behaviors": "หาชะมดและพฤติกรรม",
    "The laptop stays usable: the job runs at low priority and pauses when the GPU is hot. You can close this page; "
    "you can stop and later continue - finished hours are kept.":
        "ยังใช้เครื่องได้ตามปกติ: งานทำด้วยลำดับความสำคัญต่ำและหยุดพักเมื่อการ์ดจอร้อน ปิดหน้านี้ได้ "
        "และหยุดแล้วทำต่อภายหลังได้ ชั่วโมงที่เสร็จแล้วจะถูกเก็บไว้",
    "Start processing": "เริ่มประมวลผล",
    "Upload finished": "อัปโหลดเสร็จแล้ว",
    "No readable videos were found in these folders.": "ไม่พบวิดีโอที่อ่านได้ในโฟลเดอร์เหล่านี้",
    "Another job is running: wait for it to finish first.": "มีงานอื่นกำลังทำอยู่ รอให้เสร็จก่อน",
    # job page
    "Starting ...": "กำลังเริ่ม ...",
    "left (estimate)": "เหลือ (โดยประมาณ)",
    "running for": "ทำงานมาแล้ว",
    "state": "สถานะ",
    "Stop": "หยุด",
    "Stopping is safe: finished parts are kept and starting the same job again continues from there.":
        "หยุดได้อย่างปลอดภัย: ส่วนที่เสร็จแล้วจะถูกเก็บไว้ และเริ่มงานเดิมอีกครั้งจะทำต่อจากจุดนั้น",
    "Details (log)": "รายละเอียด (log)",
    "finished": "เสร็จแล้ว",
    "stopped with an error": "หยุดเพราะเกิดข้อผิดพลาด",
    "was stopped": "ถูกหยุด",
    "Finished.": "เสร็จแล้ว",
    "Start again (continues where it stopped)": "เริ่มอีกครั้ง (ทำต่อจากจุดที่หยุด)",
    "See the log below.": "ดู log ด้านล่าง",
    "Preparing the videos (2 frames per second)": "กำลังเตรียมวิดีโอ (2 ภาพต่อวินาที)",
    "Finding the civet in every frame": "กำลังหาชะมดในทุกภาพ",
    "Analysis done": "วิเคราะห์เสร็จแล้ว",
    "Finding short stops and making pictures": "กำลังหาช่วงหยุดนิ่งและทำภาพ",
    "Writing the sheet and charts": "กำลังเขียนตารางและกราฟ",
    "Reading the saved frames": "กำลังอ่านข้อมูลที่บันทึกไว้",
    "Cleaning and counting behaviors": "กำลังกรองและนับพฤติกรรม",
    "Cutting civet pictures from the checked stops": "กำลังตัดภาพชะมดจากช่วงที่ตรวจแล้ว",
    "Testing on stops the model never saw": "กำลังทดสอบกับช่วงที่โมเดลไม่เคยเห็น",
    "Done": "เสร็จ",
    # overview
    "Inactive (resting)": "ไม่เคลื่อนไหว (พัก)",
    "Not seen by any camera": "ไม่มีกล้องใดเห็น",
    "feeding / drinking visits": "ครั้งที่กินอาหาร / ดื่มน้ำ",
    "territorial marking · grooming · urination · defecation (checked)":
        "ทำเครื่องหมายอาณาเขต · ทำความสะอาดตัว · ปัสสาวะ · อุจจาระ (ตรวจแล้ว)",
    "What to do": "สิ่งที่ต้องทำ",
    "Short videos of every visit to the sleeping box top, every moment the civet stood still, and every moment it moved "
    "its body while staying in place. You say if it was urination, defecation, territorial marking, grooming or nothing. "
    "Only what you confirm is counted.":
        "วิดีโอสั้นของทุกครั้งที่ชะมดขึ้นไปบนหลังคากล่องนอน ทุกช่วงที่ยืนนิ่ง และทุกช่วงที่ขยับตัวอยู่กับที่ "
        "คุณเลือกว่าเป็นปัสสาวะ อุจจาระ ทำเครื่องหมายอาณาเขต ทำความสะอาดตัว หรือไม่ใช่ จะนับเฉพาะที่คุณยืนยันเท่านั้น",
    "Open Behavior check": "เปิดหน้าตรวจพฤติกรรม",
    "Check unsure feeding / drinking": "ตรวจการกินอาหาร / ดื่มน้ำที่โปรแกรมไม่แน่ใจ",
    "Watch each one and say if it was right.": "ดูแต่ละรายการแล้วบอกว่าถูกหรือไม่",
    "Open the list": "เปิดรายการ",
    "Rebuild the report": "สร้างรายงานใหม่",
    "Rebuild report": "สร้างรายงานใหม่",
    "Rebuild report now": "สร้างรายงานใหม่ตอนนี้",
    "rebuild report": "สร้างรายงานใหม่",
    "You have answers that are not in the sheet yet.": "มีคำตอบที่ยังไม่อยู่ในตาราง",
    "Teach the app": "สอนโปรแกรม",
    "(optional, after the Behavior check) - trains the behavior model on your answers (about 15 min).":
        "(ไม่บังคับ หลังตรวจพฤติกรรม) - ฝึกโมเดลพฤติกรรมด้วยคำตอบของคุณ (ประมาณ 15 นาที)",
    "Train behavior model": "ฝึกโมเดลพฤติกรรม",
    "Open Spot check": "เปิดหน้าสุ่มตรวจ",
    "What the words mean": "ความหมายของคำต่างๆ",
    # glossary
    "The civet is resting: lying down, stretched out, curled up or sleeping in a box. It only switches between Inactive "
    "and Active after 1 minute or more, so a short stretch or a few steps do not count as a change.":
        "ชะมดกำลังพัก: นอนราบ นอนเหยียดตัว นอนขดตัว หรือนอนในกล่อง จะเปลี่ยนระหว่างไม่เคลื่อนไหวกับเคลื่อนไหว"
        "ก็ต่อเมื่อนาน 1 นาทีขึ้นไป การยืดตัวสั้นๆ หรือเดินไม่กี่ก้าวจึงไม่นับว่าเปลี่ยน",
    "The civet is awake and doing something: walking, eating, drinking, sniffing, exploring.":
        "ชะมดตื่นและกำลังทำบางอย่าง: เดิน กิน ดื่ม ดม สำรวจ",
    "No camera can see the civet (for example it is sleeping in a spot outside every camera's view). "
    "This time is neither Active nor Inactive - we do not guess.":
        "ไม่มีกล้องใดเห็นชะมด (เช่น นอนอยู่ในจุดที่อยู่นอกมุมกล้องทุกตัว) "
        "ช่วงเวลานี้ไม่นับเป็นเคลื่อนไหวหรือไม่เคลื่อนไหว - โปรแกรมไม่เดา",
    "Not seen": "ไม่เห็น",
    "Same as Not visible, in the 'time per room' chart.": "เหมือนกับ มองไม่เห็น ในกราฟ 'เวลาในแต่ละห้อง'",
    "Share of the visible time that was Active. Not-visible time is left out.":
        "สัดส่วนของเวลาที่เห็นตัวซึ่งชะมดเคลื่อนไหว ไม่รวมช่วงที่มองไม่เห็น",
    "Feeding / Drinking (visits)": "กินอาหาร / ดื่มน้ำ (ครั้ง)",
    "The civet stays at the food dish / water basin with its head at it. Looking up and going back within 1 minute is "
    "the same visit; walking past the bowl does not count.":
        "ชะมดอยู่ที่ถาดอาหาร / อ่างน้ำโดยหัวอยู่ที่ถาด เงยหน้าแล้วกลับมาภายใน 1 นาทีนับเป็นครั้งเดียวกัน "
        "เดินผ่านถาดไม่นับ",
    "Licking, nibbling or scratching its own fur.": "เลีย แทะ หรือเกาขนตัวเอง",
    "Territorial marking (scent marking)": "ทำเครื่องหมายอาณาเขต (ถูกลิ่น)",
    "Rubbing or pressing its rear end - where the scent gland is - on a log, wall, door frame or fence.":
        "ถูหรือกดก้น (ตำแหน่งต่อมกลิ่น) กับขอนไม้ ผนัง กรอบประตู หรือรั้ว",
    "Urinating.": "ปัสสาวะ",
    "Defecating.": "ถ่ายอุจจาระ",
    "Short stop": "หยุดนิ่งช่วงสั้น",
    "The civet walked, stood still for 4-60 seconds, then walked on. Grooming, territorial marking, urination and "
    "defecation happen in these stops, so every one is shown to a person (Behavior check).":
        "ชะมดเดิน แล้วหยุดนิ่ง 4-60 วินาที แล้วเดินต่อ การทำความสะอาดตัว ทำเครื่องหมายอาณาเขต ปัสสาวะ และอุจจาระ "
        "เกิดในช่วงเหล่านี้ จึงให้คนตรวจทุกช่วง (ตรวจพฤติกรรม)",
    "On the sleeping box top": "บนหลังคากล่องนอน",
    "Every visit of the civet to the top of the sleeping box (camera 2), where it usually marks, urinates and defecates. "
    "Shown first in Behavior check with a before / after picture of the box top.":
        "ทุกครั้งที่ชะมดขึ้นไปบนหลังคากล่องนอน (กล้อง 2) ซึ่งเป็นที่ที่มักทำเครื่องหมาย ปัสสาวะ และอุจจาระ "
        "แสดงเป็นอันดับแรกในหน้าตรวจพฤติกรรม พร้อมภาพหลังคากล่องก่อน / หลัง",
    "A short stop in a doorway while the civet moves to the next room - not checked.":
        "หยุดสั้นๆ ที่ประตูขณะชะมดเดินไปห้องถัดไป - ไม่ต้องตรวจ",
    "Please check / hollow dot": "โปรดตรวจ / จุดกลวง",
    "The app is not sure. Look at the clip or pictures and press Correct, Not this behavior, or pick what it really was.":
        "โปรแกรมไม่แน่ใจ ดูคลิปหรือภาพแล้วกด ถูกต้อง, ไม่ใช่พฤติกรรมนี้ หรือเลือกว่าจริงๆ แล้วคืออะไร",
    "Counted / filled dot": "นับแล้ว / จุดทึบ",
    "Included in the observation sheet.": "รวมอยู่ในตารางบันทึกพฤติกรรมแล้ว",
    "Confidence": "ความมั่นใจ",
    "How sure the civet detector is that the box really is the civet (0 to 1).":
        "ตัวตรวจจับมั่นใจแค่ไหนว่ากรอบนั้นคือชะมดจริง (0 ถึง 1)",
    "night IR / day colour": "กลางคืน IR / กลางวันภาพสี",
    "Night: the camera films in black and white infrared. Day: normal colour.":
        "กลางคืน: กล้องถ่ายภาพขาวดำอินฟราเรด กลางวัน: ภาพสีปกติ",
    "The picture was blurred, too dark or frozen - not used for counting.":
        "ภาพเบลอ มืดเกินไป หรือค้าง - ไม่นำมานับ",
    "Random moments of the day that you answer without seeing the app's answer; it gives the real accuracy of the app.":
        "ช่วงเวลาสุ่มของวันที่คุณตอบโดยไม่เห็นคำตอบของโปรแกรม ใช้วัดความแม่นยำจริงของโปรแกรม",
    # sheet page
    "The same tables as": "ตารางเดียวกับ",
    "Download sheet.xlsx": "ดาวน์โหลด sheet.xlsx",
    "your newest answers are not in it yet": "คำตอบล่าสุดของคุณยังไม่อยู่ในตาราง",
    "Events": "เหตุการณ์",
    "Inactive detail": "รายละเอียดช่วงพัก",
    "Stops to check": "ช่วงที่ต้องตรวจ",
    "Summary": "สรุป",
    "Alerts": "การแจ้งเตือน",
    "Read me": "อ่านก่อน",
    "No": "ลำดับ",
    "Start": "เริ่ม",
    "Duration": "ระยะเวลา",
    "Frequency (Time)": "ความถี่ (ครั้ง)",
    "Excretion": "การขับถ่าย",
    "Total": "รวม",
    "Item": "รายการ",
    "Value": "ค่า",
    "Alert": "การแจ้งเตือน",
    "Word": "คำ",
    "Meaning": "ความหมาย",
    # charts
    "Click the picture to enlarge, click again to zoom in. Filled dots = counted, hollow dots = still to check.":
        "คลิกภาพเพื่อขยาย คลิกอีกครั้งเพื่อซูม จุดทึบ = นับแล้ว จุดกลวง = ยังต้องตรวจ",
    "Download charts.png": "ดาวน์โหลด charts.png",
    # behavior check
    "Not checked yet": "ยังไม่ได้ตรวจ",
    "All": "ทั้งหมด",
    "Sleeping box top": "หลังคากล่องนอน",
    "Stood still": "ยืนนิ่ง",
    "Moving in place": "ขยับตัวอยู่กับที่",
    "Found behaviors": "พฤติกรรมที่พบ",
    "Sorted: most likely behaviors first": "เรียงลำดับ: รายการที่น่าจะเป็นพฤติกรรมมากที่สุดก่อน",
    "Sorted: most likely behaviors first (your trained model)": "เรียงลำดับ: รายการที่น่าจะเป็นพฤติกรรมมากที่สุดก่อน (โมเดลที่คุณฝึก)",
    "zoom": "ซูม",
    "speed": "ความเร็ว",
    "4 pictures (enlarge)": "4 ภาพ (ขยาย)",
    "Nothing special": "ไม่มีอะไรพิเศษ",
    "Can't see": "มองไม่เห็น",
    "Stands or half-squats with the hindquarters lowered and the tail lifted; usually a few seconds. On the box top: "
    "look for a new wet (darker) patch in the 'after' picture.":
        "ยืนหรือย่อตัวครึ่งหนึ่ง สะโพกต่ำ ยกหาง มักไม่กี่วินาที บนหลังคากล่อง: ดูรอยเปียก (สีเข้มขึ้น) ใหม่ในภาพ 'หลัง'",
    "Deep squat, back hunched, tail raised, held still longer (often 5 s or more). On the box top: look for new droppings "
    "(marked red) in the 'after' picture.":
        "ย่อตัวลึก หลังโก่ง ยกหาง นิ่งนานกว่า (มักตั้งแต่ 5 วินาที) บนหลังคากล่อง: ดูมูลใหม่ (ทำเครื่องหมายสีแดง) ในภาพ 'หลัง'",
    "Presses or rubs its rear end (scent gland under the tail) on the box, a log, wall, door frame or fence.":
        "กดหรือถูก้น (ต่อมกลิ่นใต้โคนหาง) กับกล่อง ขอนไม้ ผนัง กรอบประตู หรือรั้ว",
    "Licks, nibbles or scratches its own fur; head turned back to its body or a hind leg scratching.":
        "เลีย แทะ หรือเกาขนตัวเอง หันหัวกลับมาที่ลำตัว หรือใช้ขาหลังเกา",
    "Lying down": "นอนราบ", "Stretched out": "นอนเหยียดตัว", "Curled up / sleeping": "นอนขดตัว / หลับ",
    "Sitting / standing": "นั่ง / ยืน",
    "5 · Lying down": "5 · นอนราบ", "6 · Stretched out": "6 · นอนเหยียดตัว",
    "7 · Curled up / sleeping": "7 · นอนขดตัว / หลับ", "8 · Sitting / standing": "8 · นั่ง / ยืน",
    "Resting or napping on its belly or side, head down or on its paws (Inactive).":
        "พักหรืองีบ นอนคว่ำหรือนอนตะแคง หัววางลงหรือวางบนขาหน้า (ไม่เคลื่อนไหว)",
    "Lying with the body and legs stretched long (Inactive).": "นอนโดยยืดลำตัวและขาออกยาว (ไม่เคลื่อนไหว)",
    "Curled into a ball, tail around the body, usually asleep (Inactive).": "นอนขดเป็นวง หางพันรอบตัว มักหลับอยู่ (ไม่เคลื่อนไหว)",
    "Awake and staying in place: sitting, standing, looking around (Active).": "ตื่นและอยู่กับที่: นั่ง ยืน มองไปรอบๆ (เคลื่อนไหว)",
    "Walking, sniffing, turning or anything else - not one of the behaviors above.":
        "เดิน ดม หมุนตัว หรืออย่างอื่น - ไม่ใช่พฤติกรรมข้างต้น",
    "Standing, sniffing, looking around, sitting, resting or turning - not one of the behaviors above.":
        "ยืน ดม มองไปรอบๆ นั่ง พัก หรือหมุนตัว - ไม่ใช่พฤติกรรมข้างต้น",
    "The civet is hidden or the picture is too dark: skipped, not used for training.":
        "ชะมดถูกบังหรือภาพมืดเกินไป: ข้ามไป ไม่ใช้ฝึกโมเดล",
    "previous / next ·": "ก่อนหน้า / ถัดไป ·",
    "5 s back / forward ·": "ย้อน / ไปข้างหน้า 5 วินาที ·",
    "play / pause ·": "เล่น / หยุด ·",
    "back to the event · the answer is saved at once and the next one opens.":
        "กลับไปที่เหตุการณ์ · คำตอบบันทึกทันทีและเปิดรายการถัดไปให้",
    "How to tell them apart": "วิธีแยกแต่ละพฤติกรรม",
    "Nothing to check. Rebuild the report first.": "ไม่มีรายการให้ตรวจ สร้างรายงานใหม่ก่อน",
    # video player
    "−5 s": "−5 วิ",
    "+5 s": "+5 วิ",
    "↺ Back to the event": "↺ กลับไปที่เหตุการณ์",
    "loop the event": "เล่นวนเหตุการณ์",
    "orange = the event · drag the bar to see before / after": "สีส้ม = เหตุการณ์ · ลากแถบเพื่อดูก่อน / หลัง",
    "Zoom on civet": "ซูมที่ชะมด",
    "Whole picture": "ทั้งภาพ",
    "Speed": "ความเร็ว",
    "labelled clip": "คลิปที่มีป้าย",
    "4 pictures": "4 ภาพ",
    "No camera saw the civet here": "ไม่มีกล้องใดเห็นชะมดในช่วงนี้",
    "Video not found": "ไม่พบวิดีโอ",
    "No video at this time": "ไม่มีวิดีโอในเวลานี้",
    # events page
    "Click any row to watch that moment from the camera (": "คลิกแถวใดก็ได้เพื่อดูช่วงนั้นจากกล้อง (",
    "next event,": "เหตุการณ์ถัดไป,",
    "5 s back / forward,": "ย้อน / ไปข้างหน้า 5 วินาที,",
    "play / pause). If the app was wrong, correct it on the right; then rebuild the report.":
        "เล่น / หยุด) ถ้าโปรแกรมผิด แก้ไขได้ทางขวา แล้วสร้างรายงานใหม่",
    "Counted (feeding, drinking ...)": "นับแล้ว (กินอาหาร, ดื่มน้ำ ...)",
    "Resting / walking": "พัก / เดิน",
    "Please check": "โปรดตรวจ",
    "please check": "โปรดตรวจ",
    "Checked by you": "ตรวจโดยคุณ",
    "Short stops": "ช่วงหยุดนิ่ง",
    "Removed": "ถูกตัดออก",
    "search (time, behavior, room)": "ค้นหา (เวลา, พฤติกรรม, ห้อง)",
    "Length": "ระยะเวลา",
    "Behavior": "พฤติกรรม",
    "Room": "ห้อง",
    "Sure": "มั่นใจ",
    "Pick an event on the left.": "เลือกเหตุการณ์ทางซ้าย",
    "✓ Correct": "✓ ถูกต้อง",
    "✗ Not this behavior": "✗ ไม่ใช่พฤติกรรมนี้",
    "It was …": "จริงๆ แล้วคือ …",
    "correct": "ถูกต้อง",
    "not this behavior": "ไม่ใช่พฤติกรรมนี้",
    "Save failed": "บันทึกไม่สำเร็จ",
    "Answer short stops in": "ตอบช่วงหยุดนิ่งได้ที่",
    # spot check
    "results": "ผลลัพธ์",
    "All moments are checked.": "ตรวจครบทุกช่วงแล้ว",
    "See the results.": "ดูผลลัพธ์",
    "Spot check results": "ผลการสุ่มตรวจ",
    "continue checking": "ตรวจต่อ",
    "checked": "ตรวจแล้ว",
    "found when in view": "หาเจอเมื่ออยู่ในกล้อง",
    "no false civet when out of view": "ไม่เห็นชะมดผิดเมื่อไม่อยู่ในกล้อง",
    "right room": "ห้องถูกต้อง",
    "Inactive/Active right": "ไม่เคลื่อนไหว/เคลื่อนไหว ถูกต้อง",
    "behavior right": "พฤติกรรมถูกต้อง",
    "Behavior: checker vs CivetWatch": "พฤติกรรม: ผู้ตรวจ เทียบกับ CivetWatch",
    "CivetWatch said": "CivetWatch ตอบ",
    "checker said": "ผู้ตรวจตอบ",
    "Disagreements": "รายการที่ไม่ตรงกัน",
    "look again": "ดูอีกครั้ง",
    "1 · Where is the civet at t?": "1 · ชะมดอยู่ที่ไหนในเวลา t?",
    "Tick every room you see it in - near the R2 / R3 opening it can show in both cameras at once.":
        "เลือกทุกห้องที่เห็นชะมด - ใกล้ช่องระหว่าง R2 / R3 อาจเห็นในสองกล้องพร้อมกัน",
    "Not visible in any camera": "ไม่เห็นในกล้องใดเลย",
    "2 · What is it doing?": "2 · ชะมดกำลังทำอะไร?",
    "(skip if not visible)": "(ข้ามได้ถ้ามองไม่เห็น)",
    "your name": "ชื่อของคุณ",
    "Save and next": "บันทึกและไปต่อ",
    "Can't tell": "บอกไม่ได้",
    "Make the spot check (about 5 min)": "สร้างชุดสุ่มตรวจ (ประมาณ 5 นาที)",
    # files
    "Everything is in": "ไฟล์ทั้งหมดอยู่ที่",
    "Open the folder": "เปิดโฟลเดอร์",
    "File": "ไฟล์",
    "What it is": "คืออะไร",
    "Size": "ขนาด",
    "download": "ดาวน์โหลด",
    "Day timeline, time per room, hourly counts, time budget": "ไทม์ไลน์ทั้งวัน เวลาในแต่ละห้อง จำนวนรายชั่วโมง สัดส่วนเวลา",
    "Every event with start, stop, room, camera": "ทุกเหตุการณ์ พร้อมเวลาเริ่ม เวลาสิ้นสุด ห้อง กล้อง",
    "The app's answer for every half second (large)": "คำตอบของโปรแกรมทุกครึ่งวินาที (ไฟล์ใหญ่)",
    "Your answers (Behavior check, corrections)": "คำตอบของคุณ (ตรวจพฤติกรรม, การแก้ไข)",
    "Observation sheet for Excel (zoo layout, Read me tab)": "ตารางบันทึกพฤติกรรมสำหรับ Excel (รูปแบบของสวนสัตว์ มีแท็บอ่านก่อน)",
    "Day totals": "ยอดรวมของวัน",
    "Test result of the trained behavior model": "ผลทดสอบโมเดลพฤติกรรมที่ฝึกแล้ว",
    "Behavior model answers on the test stops": "คำตอบของโมเดลพฤติกรรมในช่วงทดสอบ",
    "1 · Urination": "1 · ปัสสาวะ", "2 · Defecation": "2 · อุจจาระ",
    "3 · Territorial marking": "3 · ทำเครื่องหมายอาณาเขต", "4 · Grooming": "4 · ทำความสะอาดตัว",
    "0 · Nothing special": "0 · ไม่มีอะไรพิเศษ", "9 · Can't see": "9 · มองไม่เห็น",
    # spot check results table (raw columns and labels)
    "id": "ลำดับ", "t": "เวลา", "true_room": "ห้อง (ผู้ตรวจ)", "pred_room": "ห้อง (โปรแกรม)", "room_ok": "ห้องถูก",
    "true_label": "พฤติกรรม (ผู้ตรวจ)", "pred_label": "พฤติกรรม (โปรแกรม)", "pred_state": "สถานะ (โปรแกรม)",
    "link": "ลิงก์", "True": "ใช่", "False": "ไม่ใช่", "none": "ไม่มี", "feeding": "กินอาหาร", "drinking": "ดื่มน้ำ",
    "grooming": "ทำความสะอาดตัว", "territorial_marking": "ทำเครื่องหมายอาณาเขต", "urination": "ปัสสาวะ",
    "defecation": "อุจจาระ", "walking": "เดิน", "active_other": "เคลื่อนไหวอื่นๆ", "lying_down": "นอนราบ",
    "curling_up": "นอนขดตัว", "stretched_out": "นอนเหยียดตัว", "sleep_in_box": "นอนในกล่อง", "cant_tell": "บอกไม่ได้",
    "not_visible": "มองไม่เห็น",
    # server messages
    "Another job is still running. Wait for it or cancel it first.": "มีงานอื่นกำลังทำอยู่ รอให้เสร็จหรือหยุดก่อน",
    "A job for this day is still running. Stop it first.": "งานของวันนี้ยังทำอยู่ หยุดงานก่อน",
}

# text with numbers / names: [regex, Thai replacement] (JavaScript regex syntax)
PATTERNS = [
    [r"^Camera (\d)$", "กล้อง $1"],
    [r"^\(optional\) - random moments you answer without seeing the app's answer; gives the real accuracy\. (\d+) of (\d+) moments checked\.$",
     "(ไม่บังคับ) - ช่วงเวลาสุ่มที่คุณตอบโดยไม่เห็นคำตอบของโปรแกรม ใช้วัดความแม่นยำจริง ตรวจแล้ว $1 จาก $2 ช่วง"],
    [r"^stood still (\d+) s$", "ยืนนิ่ง $1 วินาที"],
    [r"^moving in place (\d+) s$", "ขยับตัวอยู่กับที่ $1 วินาที"],
    [r"^on the sleeping box top (\d+) s$", "บนหลังคากล่องนอน $1 วินาที"],
    [r"^passing a door, still (\d+) s$", "หยุดที่ประตู $1 วินาที"],
    [r"^active (.+) · inactive (.+)$", "เคลื่อนไหว $1 · ไม่เคลื่อนไหว $2"],
    [r"^(\d+) feeding · (\d+) drinking visits$", "กินอาหาร $1 ครั้ง · ดื่มน้ำ $2 ครั้ง"],
    [r"^(\d+) to check$", "เหลือตรวจ $1"],
    [r"^- (\d+) of (\d+) left$", "- เหลือ $1 จาก $2"],
    [r"^- (\d+) left$", "- เหลือ $1"],
    [r"^of (\d+) checked$", "จาก $1 รายการตรวจแล้ว"],
    [r"^· (\d+) of (\d+) checked ·$", "· ตรวจแล้ว $1 จาก $2 ·"],
    [r"^Active · (\d+)% of the time it was seen$", "เคลื่อนไหว · $1% ของเวลาที่เห็นตัว"],
    [r"^Active (\d+) % of the time it was seen$", "เคลื่อนไหว $1% ของเวลาที่เห็นตัว"],
    [r"^so your answers are in the sheet and charts \(about (.+)\)\.$", "เพื่อให้คำตอบของคุณอยู่ในตารางและกราฟ (ประมาณ $1)"],
    [r"^\(optional\) - random moments you answer without seeing the app's answer; gives the real accuracy\.(.*)$",
     "(ไม่บังคับ) - ช่วงเวลาสุ่มที่คุณตอบโดยไม่เห็นคำตอบของโปรแกรม ใช้วัดความแม่นยำจริง$1"],
    [r"(\d+) of (\d+) moments checked\.", "ตรวจแล้ว $1 จาก $2 ช่วง"],
    [r"^Civet not visible for (\d+) min in total, please check the cameras$", "ไม่เห็นชะมดรวม $1 นาที โปรดตรวจสอบกล้อง"],
    [r"^No feeding after (.+), please check$", "ไม่มีการกินอาหารหลัง $1 โปรดตรวจสอบ"],
    [r"^(\d+)% of frames had a poor image \(blur / dark / frozen\)$", "$1% ของภาพไม่ชัด (เบลอ / มืด / ค้าง)"],
    [r"^Last training: (.+)$", "ฝึกครั้งล่าสุด: $1"],
    [r"^Analysing (\S+) \(part (\d+) of (\d+)\)$", "กำลังวิเคราะห์ $1 (ส่วนที่ $2 จาก $3)"],
    [r"^Preparing (\w+): (.+)$", "กำลังเตรียม $1: $2"],
    [r"^Training the behavior model \(round (\d+) of (\d+)\)$", "กำลังฝึกโมเดลพฤติกรรม (รอบ $1 จาก $2)"],
    [r"^Rebuild report (.+)$", "สร้างรายงานใหม่ $1"],
    [r"^Train behavior model \((.+)\)$", "ฝึกโมเดลพฤติกรรม ($1)"],
    [r"^Make spot check \((.+)\)$", "สร้างชุดสุ่มตรวจ ($1)"],
    [r"^Process (\d{4}-\d\d-\d\d)$", "ประมวลผล $1"],
    [r"^Open (\d{4}-\d\d-\d\d)$", "เปิด $1"],
    [r"^The job (.+)\.$", "งาน$1"],
    [r"^Moment (\d+) · (\S+) · rows = cameras / rooms, columns = 1 s before, the moment \(t\), 1 s after\. "
     r"The app's answer stays hidden\.$",
     "ช่วงที่ $1 · $2 · แถว = กล้อง / ห้อง, คอลัมน์ = 1 วินาทีก่อน, ช่วงนั้น (t), 1 วินาทีหลัง คำตอบของโปรแกรมถูกซ่อนไว้"],
    [r"^Uploading (.+) · (.+) of (.+) GB$", "กำลังอัปโหลด $1 · $2 จาก $3 GB"],
    [r"^(.+) GB free$", "ว่าง $1 GB"],
    [r"^disk needed · (.+) GB free$", "พื้นที่ที่ต้องใช้ · ว่าง $1 GB"],
    [r"^Saved: (.+)$", "บันทึกแล้ว: $1"],
    [r"^it was (.+)$", "จริงๆ คือ $1"],
    [r"^checked by Claude \(AI\)(.*)$", "ตรวจโดย Claude (AI)$1"],
    [r"^checked by a person(.*)$", "ตรวจโดยคน$1"],
    [r"^removed by a person(.*)$", "ถูกตัดออกโดยคน$1"],
]

# pieces replaced inside any remaining text (after the patterns), longest first
PIECES = {
    "Sunday": "วันอาทิตย์", "Monday": "วันจันทร์", "Tuesday": "วันอังคาร", "Wednesday": "วันพุธ",
    "Thursday": "วันพฤหัสบดี", "Friday": "วันศุกร์", "Saturday": "วันเสาร์",
    "January": "มกราคม", "February": "กุมภาพันธ์", "March": "มีนาคม", "April": "เมษายน", "May": "พฤษภาคม",
    "June": "มิถุนายน", "July": "กรกฎาคม", "August": "สิงหาคม", "September": "กันยายน", "October": "ตุลาคม",
    "November": "พฤศจิกายน", "December": "ธันวาคม",
    "Jan": "ม.ค.", "Feb": "ก.พ.", "Mar": "มี.ค.", "Apr": "เม.ย.", "Jun": "มิ.ย.", "Jul": "ก.ค.", "Aug": "ส.ค.",
    "Sep": "ก.ย.", "Oct": "ต.ค.", "Nov": "พ.ย.", "Dec": "ธ.ค.",
    "stood still": "ยืนนิ่ง", "moving in place": "ขยับตัวอยู่กับที่", "on the sleeping box top": "บนหลังคากล่องนอน",
    "passing a door, still": "หยุดที่ประตู", "box top changed": "หลังคากล่องเปลี่ยนไป",
    "(red in the 'after' picture)": "(สีแดงในภาพ 'หลัง')", "app's guess:": "โปรแกรมเดา:",
    "your answer:": "คำตอบของคุณ:", "app:": "โปรแกรม:", "no camera": "ไม่มีกล้อง",
    "night ir": "กลางคืน IR", "day colour": "กลางวัน", "visible": "มองเห็น", "out of view": "นอกกล้อง",
    "sure": "มั่นใจ", "left": "เหลือ", "is running": "กำลังทำงาน",
    "(rebuild the report to update the sheet)": "(สร้างรายงานใหม่เพื่ออัปเดตตาราง)",
    " - new answer, rebuild the report to update the sheet": " - คำตอบใหม่ สร้างรายงานใหม่เพื่ออัปเดตตาราง",
    " min": " นาที", " h ": " ชม. ",
}

# whole dialogs (alert / confirm): English start -> Thai
DIALOGS = {
    "Pick the video files first": "เลือกไฟล์วิดีโอก่อน",
    "Stop this job? Finished parts are kept.": "หยุดงานนี้? ส่วนที่เสร็จแล้วจะถูกเก็บไว้",
    "Tick where the civet is (or \"Not visible\").": "เลือกว่าชะมดอยู่ที่ไหน (หรือ \"ไม่เห็นในกล้องใดเลย\")",
    "Could not start": "เริ่มงานไม่ได้",
    "Could not remove": "ลบไม่ได้",
}


def script():
    """<script> for the base page: the switch, and the translation of the page."""
    data = json.dumps({"exact": EXACT, "patterns": PATTERNS,
                       "pieces": sorted(PIECES.items(), key=lambda kv: -len(kv[0])), "dialogs": DIALOGS},
                      ensure_ascii=False)
    return "<script>const CW_I18N=" + data + ";\n" + _JS + "</script>"


_JS = r"""
let CW_LANG='en';try{CW_LANG=localStorage.getItem('cw_lang')||'en';}catch(e){}
function L(en,th){return CW_LANG==='th'?th:en;}
function cwTr(s){
  const t=s.trim().replace(/\s+/g,' ');if(!t||!/[A-Za-z]{2}/.test(t))return s;
  const lead=s.match(/^\s*/)[0],tail=s.match(/\s*$/)[0];
  if(CW_I18N.exact[t]!==undefined)return lead+CW_I18N.exact[t]+tail;
  for(const [p,r] of CW_I18N.patterns){const re=new RegExp(p);if(re.test(t)){
    let o=cwPieces(t.replace(re,r));return lead+o.replace(/\b(Inactive|Active|Not visible|Feeding|Drinking|Territorial marking|Grooming|Urination|Defecation|Locomotion \/ Walking|Lying down|Curling up|Stretched out|Active \(other\))\b/g,w=>CW_I18N.exact[w]||w)+tail;}}
  return lead+cwPieces(t)+tail;}
function cwPieces(o){for(const [a,b] of CW_I18N.pieces){if(!o.includes(a))continue;
    const k=a.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');o=o.replace(new RegExp('(?<![A-Za-z])'+k+'(?![A-Za-z])','g'),b);}
  return o.replace(/(\d+(?:\.\d+)?) s(?![A-Za-z])/g,'$1 วินาที');}
const cwDone=new WeakMap();
function cwNode(n){
  if(n.nodeType===3){const p=n.parentElement;if(!p||p.closest('script,style,code,.notr,kbd,pre'))return;
    if(cwDone.get(n)===n.data)return;const o=cwTr(n.data);if(o!==n.data)n.data=o;cwDone.set(n,n.data);return;}
  if(n.nodeType!==1||n.matches('script,style,code,.notr,pre'))return;
  for(const a of ['placeholder','title']){const v=n.getAttribute&&n.getAttribute(a);if(v){const o=cwTr(v);if(o!==v)n.setAttribute(a,o);}}
  if(n.tagName==='OPTION'||n.tagName==='BUTTON'||n.childNodes.length)n.childNodes.forEach(cwNode);}
function cwApply(){document.documentElement.lang='th';cwNode(document.body);document.title=cwTr(document.title);
  new MutationObserver(ms=>{for(const m of ms){if(m.type==='characterData')cwNode(m.target);else m.addedNodes.forEach(cwNode);}})
    .observe(document.body,{childList:true,subtree:true,characterData:true});}
function cwDialog(m){m=String(m);for(const [a,b] of Object.entries(CW_I18N.dialogs)){if(m.startsWith(a))return b;}
  if(CW_I18N.exact[m.trim()])return CW_I18N.exact[m.trim()];return m;}
function cwSetLang(l){try{localStorage.setItem('cw_lang',l);}catch(e){}location.reload();}
if(CW_LANG==='th'){const A=window.alert.bind(window),C=window.confirm.bind(window);
  window.alert=m=>A(cwDialog(m));window.confirm=m=>C(cwDialog(m));
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',cwApply);else cwApply();}
document.addEventListener('DOMContentLoaded',()=>{const b=document.getElementById('langbtn');if(b){
  b.textContent=CW_LANG==='th'?'EN':'ไทย';b.title=CW_LANG==='th'?'Switch to English':'เปลี่ยนเป็นภาษาไทย';
  b.onclick=()=>cwSetLang(CW_LANG==='th'?'en':'th');}});
"""
