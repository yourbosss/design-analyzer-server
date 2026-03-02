import os
import re
import time
import json
from PIL import Image, ImageDraw, ImageFont
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import cv2
import numpy as np


def take_screenshot(url, save_path="screenshot.png"):
    """
    Делает скриншот страницы и сохраняет по указанному пути.
    """
    print(f"📸 Делаем скриншот: {url}")
    chrome_driver_path = r"D:\unik\diplom\design_analyzer_server\chromedriver.exe"
    
    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service)
    
    driver.get(url)
    time.sleep(5)  # Ждём загрузки страницы
    
    driver.save_screenshot(save_path)
    driver.quit()
    
    print(f"✅ Скриншот сохранён: {save_path}")
    return save_path


def call_florence2(screenshot_path):
    """
    Использует Florence-2 для детекции UI-элементов на скриншоте.
    Возвращает JSON с найденными элементами.
    """
    print(f"Анализируем через Florence-2: {screenshot_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True
    )
    
    image = Image.open(screenshot_path).convert("RGB")
    
    prompt = "<OD>"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height)
    )
    
    print(f"✅ Ответ модели: {parsed_answer}")
    
    # =======================================
    # Парсим ответ Florence-2
    # =======================================
    elements = {
        "buttons": [],
        "headings": [],
        "text_blocks": [],
        "images": [],
        "links": [],
        "input_fields": []
    }
    
    if "<OD>" in parsed_answer:
        od_data = parsed_answer["<OD>"]
        bboxes = od_data.get("bboxes", [])
        labels = od_data.get("labels", [])
        
        for bbox, label in zip(bboxes, labels):
            # Определяем тип элемента по метке
            label_lower = label.lower()
            element = {
                "text": label,
                "bbox": bbox,
                "type": "ui_element"
            }
            
            if "button" in label_lower or "btn" in label_lower:
                element["type"] = "button"
                elements["buttons"].append(element)
            elif "heading" in label_lower or "title" in label_lower or "h1" in label_lower:
                element["type"] = "heading"
                elements["headings"].append(element)
            elif "image" in label_lower or "img" in label_lower or "photo" in label_lower:
                element["type"] = "image"
                elements["images"].append(element)
            elif "link" in label_lower or "a href" in label_lower:
                element["type"] = "link"
                elements["links"].append(element)
            elif "input" in label_lower or "field" in label_lower or "search" in label_lower:
                element["type"] = "input_field"
                elements["input_fields"].append(element)
            else:
                element["type"] = "text_block"
                elements["text_blocks"].append(element)
        
        print(f"✅ Найдено элементов: {len(bboxes)}")
    else:
        print("⚠️ Модель не нашла элементов. Добавляю тестовый для демонстрации.")
        elements["text_blocks"].append({
            "text": "demo_element",
            "bbox": [100, 100, 200, 200],
            "type": "text_block"
        })
    
    return elements


# =======================================
# ЭТАП 3: Извлечение цветов
# =======================================
def extract_colors(elements, screenshot_path):
    """
    По координатам элементов извлекает реальные цвета из скриншота.
    """
    print(f"🎨 Извлекаем цвета из {screenshot_path}")
    
    try:
        img = Image.open(screenshot_path)
        
        # Обрабатываем все категории элементов
        all_elements = []
        for category in ["buttons", "headings", "text_blocks", "images", "links", "input_fields"]:
            all_elements.extend(elements.get(category, []))
        
        for element in all_elements:
            bbox = element["bbox"]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            element["color_text"] = img.getpixel((center_x, center_y))
            
            bg_x = bbox[0] + 5
            bg_y = bbox[1] + 5
            element["color_bg"] = img.getpixel((bg_x, bg_y))
            
            element["width"] = bbox[2] - bbox[0]
            element["height"] = bbox[3] - bbox[1]
        
        print(f"✅ Цвета извлечены для {len(all_elements)} элементов")
    except Exception as e:
        print(f"⚠️ Не удалось извлечь цвета: {e}")
    
    return elements


# =======================================
# Функция расчёта контраста (WCAG)
# =======================================
def check_contrast(rgb_text, rgb_bg):
    """
    Рассчитывает контраст между двумя RGB-цветами по формуле WCAG 2.1.
    Возвращает коэффициент контрастности.
    """
    def luminance(rgb):
        r, g, b = [x / 255.0 for x in rgb]
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    L1 = luminance(rgb_text)
    L2 = luminance(rgb_bg)
    return (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)


# =======================================
# ЭТАП 5: Проверка объективных правил
# =======================================
def check_objective_rules(elements):
    """
    Проверяет объективные критерии качества дизайна:
    - контрастность текста (уже посчитана)
    - размер кликабельных элементов
    - наличие обязательных элементов
    - цветовая палитра
    """
    print("📊 Проверяем объективные правила...")
    
    violations = []
    
    # =======================================
    # 1. Проверка размеров кликабельных элементов
    # =======================================
    min_button_size = 44  # WCAG recommendation
    
    for category in ["buttons", "links", "input_fields"]:
        for element in elements.get(category, []):
            width = element.get("width", 0)
            height = element.get("height", 0)
            
            if width < min_button_size or height < min_button_size:
                violations.append({
                    "type": "size_too_small",
                    "element_type": category,
                    "element_text": element.get("text", "unknown"),
                    "bbox": element.get("bbox", []),
                    "current_size": f"{width}×{height}",
                    "required_size": f"{min_button_size}×{min_button_size}",
                    "severity": "important"
                })
    
    # =======================================
    # 2. Проверка наличия обязательных элементов
    # =======================================
    # Проверяем наличие заголовков
    headings_count = len(elements.get("headings", []))
    if headings_count == 0:
        violations.append({
            "type": "missing_h1",
            "description": "Отсутствуют заголовки на странице",
            "severity": "critical"
        })
    
    # Проверяем наличие CTA (кнопок или ссылок)
    buttons_count = len(elements.get("buttons", []))
    links_count = len(elements.get("links", []))
    
    if buttons_count == 0 and links_count == 0:
        violations.append({
            "type": "missing_cta",
            "description": "Отсутствуют кликабельные элементы (кнопки/ссылки)",
            "severity": "critical"
        })
    
    # =======================================
    # 3. Проверка контрастности (дополнительная)
    # =======================================
    low_contrast_elements = []
    
    for category in ["buttons", "headings", "text_blocks", "links", "input_fields"]:
        for element in elements.get(category, []):
            if not element.get("contrast_ok", True):
                low_contrast_elements.append({
                    "element_type": category,
                    "element_text": element.get("text", "unknown"),
                    "contrast": element.get("contrast", 0),
                    "bbox": element.get("bbox", [])
                })
    
    if low_contrast_elements:
        violations.append({
            "type": "low_contrast_issues",
            "count": len(low_contrast_elements),
            "elements": low_contrast_elements[:5],  # первые 5 для примера
            "severity": "critical"
        })
    
    # =======================================
    # 4. Проверка цветовой палитры
    # =======================================
    all_colors = set()
    for category in ["buttons", "headings", "text_blocks", "links", "input_fields", "images"]:
        for element in elements.get(category, []):
            if "color_text" in element:
                all_colors.add(tuple(element["color_text"]))
            if "color_bg" in element:
                all_colors.add(tuple(element["color_bg"]))
    
    if len(all_colors) > 7:  # если слишком много цветов
        violations.append({
            "type": "too_many_colors",
            "color_count": len(all_colors),
            "recommendation": "Рекомендуется использовать не более 5-7 основных цветов",
            "severity": "recommendation"
        })
    
    print(f"✅ Найдено нарушений: {len(violations)}")
    return violations


# =======================================
# Функция для отрисовки bounding boxes
# =======================================
def draw_boxes(image_path, elements, output_path="result_with_boxes.png"):
    """
    Рисует прямоугольники на изображении по координатам из elements
    """
    print(f"🎨 Рисуем рамки на изображении: {output_path}")
    
    # Загружаем изображение
    img = cv2.imread(image_path)
    
    # Рисуем рамки для всех найденных элементов
    for category in ["buttons", "headings", "text_blocks", "images", "links", "input_fields"]:
        for element in elements.get(category, []):
            bbox = element["bbox"]
            # Координаты: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Разные цвета для разных типов элементов
            if category == "buttons":
                color = (0, 255, 0)  # зелёный
            elif category == "headings":
                color = (255, 0, 0)  # синий
            elif category == "images":
                color = (0, 0, 255)  # красный
            elif category == "links":
                color = (255, 255, 0)  # жёлтый
            elif category == "input_fields":
                color = (255, 0, 255)  # розовый
            else:
                color = (255, 255, 255)  # белый
            
            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Подписываем тип элемента
            cv2.putText(img, category, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Сохраняем результат
    cv2.imwrite(output_path, img)
    print(f"✅ Изображение с рамками сохранено: {output_path}")
    return output_path


# =======================================
# ГЛАВНАЯ ФУНКЦИЯ КОНВЕЙЕРА
# =======================================
def analyze_site(url):
    """
    Запускает полный цикл анализа.
    """
    print(f"\n🚀 Начинаем анализ: {url}")
    
    # Этап 1
    screenshot_path = take_screenshot(url)
    
    # Этап 2 (реальная Florence-2)
    elements = call_florence2(screenshot_path)
    
    # Этап 3
    enriched = extract_colors(elements, screenshot_path)
    
    # Проверка контраста для текстовых элементов
    print("🔍 Проверяем контраст...")
    all_elements = []
    for category in ["buttons", "headings", "text_blocks", "links", "input_fields"]:
        all_elements.extend(enriched.get(category, []))
    
    for element in all_elements:
        if "color_text" in element and "color_bg" in element:
            contrast = check_contrast(element["color_text"], element["color_bg"])
            element["contrast"] = round(contrast, 2)
            element["contrast_ok"] = contrast >= 4.5
    
    # ЭТАП 5: Проверка объективных правил
    objective_violations = check_objective_rules(enriched)
    
    # Отрисовка bounding boxes на изображении
    boxes_image_path = draw_boxes(screenshot_path, enriched, "result_with_boxes.png")
    
    # Формируем итоговый результат
    result = {
        "url": url,
        "elements": enriched,
        "objective_violations": objective_violations,
        "message": "Анализ выполнен (Florence-2)",
        "visualization": boxes_image_path
    }
    
    print(f"✅ Анализ завершён. Результат сохранён.")
    return result


# =======================================
# ТЕСТ
# =======================================
if __name__ == "__main__":
    test_url = "D:\пример!!.png"
    res = analyze_site(test_url)
    print("\n📦 Результат:")
    print(json.dumps(res, indent=2, ensure_ascii=False))