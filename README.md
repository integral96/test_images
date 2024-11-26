Программа сравнивает фотографии шахмат с разными светотенями, расфокусировкой и дает результат положительного соответствия(не зависимо от изменения угла обзора камеры, расфокусировк, и источника света)
главное что бы количеество шахматных фигур не менялось,
но стоит нам убрать одну шахматную фигуру("frame_10004_new.png") как костяк нарушается и уже нет соответствия.

Логика с множествами нарезок по градиенту:
Можества с нарезками где изменяется угол обзора , угол падения света и расфокус.
# frame_10001 = {slice_10001_01, slice_10001_02, ..., slice_10001_17}
# frame_10002 = {slice_10002_01, slice_10002_02, ..., slice_10002_17}
# frame_10003 = {slice_10003_01, slice_10003_02, ..., slice_10003_17}
# frame_10004 = {slice_10004_01, slice_10004_02, ..., slice_10004_17}

Множество нарезок без одной фигуры:
# frame_10004_new = {slice_10004_new_01, slice_10004_new_02, ..., slice_10004_new_17}

Логика:
# frame_10001 <--------> frame_10002
# frame_10001 <--------> frame_10003
# frame_10002 <--------> frame_10004
# frame_10003 <--------> frame_10004
# frame_10004 <---\\---> frame_10004_new
#
# cd test_images
# mkdir buld && cd build
# cp ../*.png .
# cmake ..
# make
