import cv2
import numpy as np

# Определите диапазон цветов для сегментации мячика (HSV)
# Например, для оранжевого мячика:
lower_color = np.array([5, 100, 100])
upper_color = np.array([15, 255, 255])


# Функция для нахождения координат мячика
def find_ball_coordinates(frame):
    # Преобразование изображения в HSV цветовое пространство
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создание маски для выделения мячика по цвету
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Применение морфологических операций для удаления шума
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Поиск контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры найдены
    if contours:
        # Находим контур с наибольшей площадью
        largest_contour = max(contours, key=cv2.contourArea)

        # Находим окружность, описывающую контур
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Условие для отсеивания слишком маленьких объектов
        if radius > 10:
            # Рисуем окружность и центр
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            return (int(x), int(y)), radius

    return None, None


# Захват видео с камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра из видео
    ret, frame = cap.read()

    if not ret:
        break

    # Нахождение координат мячика
    coordinates, radius = find_ball_coordinates(frame)

    if coordinates:
        x, y = coordinates
        # Вывод координат мячика на экран
        cv2.putText(frame, f"Coordinates: ({}, {})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Отображение кадра
    cv2.imshow('Ball Detection', frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
