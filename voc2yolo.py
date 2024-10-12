from pylabel import importer

dataset = importer.ImportVOC(path=r"C:\Users\Kenny\PycharmProjects\yolo_bear\train\label_voc")
dataset.export.ExportToYoloV5()