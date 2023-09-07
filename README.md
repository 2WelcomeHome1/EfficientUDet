# EfficientUDet+

EfficientUDet - it is an algoritm, which was development for instanse anf semantic segmentation.

Данная работа посвящена проблеме потери пространственной информации и контекста окружающей среды, возникающей при решении задачи сегментации методами машинного зрения. Проанализированы концепции актуальных разработок в области семантической сегментации изображений. Наглядно проиллюстрирована и подробно описана архитектура разработанного алгоритма семантической сегментации EfficientUDet. Представлены результаты тестирования разработанной модели в симуляторе тестирования систем автономного вождения CARLA. В результате сделан вывод, что разработанная модель не является самой точной среди ныне существующих, однако имеет значительный потенциал.

Разработка архитектуры семантической сегментации для полного понимания сцены окружающей среды в интеллектуальных транспортных системах является очень сложной задачей. В данной работе предложен новый подход к сегментации на уровне пикселей с использованием пирамидальной нейросети в качестве промежуточного блока. В данном исследовании предложено использовать алгоритмы семейства EfficientNet в качестве энкодера, а в качестве декодера использовать архитектуру UNet, который объединяет как функции высокого уровня, так и пространственную информацию низкого уровня для точной сегментации. Дополнительно на этапе «Skip-connections» был введен промежуточный блок, представляющий собой архитектуру wBiFPN, что позволяет разбивать полученное c камеры изображение на несколько фрагментов разного разрешения и масштаба с целью извлечения уникальных признаков объектов, расположенных на изображении


Тестирование и обучение алгоритма проводилось на ПК с процессором AMD 3600X, видеокартой Nvidia GTX 970 и оперативной памятью 16 ГБ. Обучение выполнялось в несколько этапов и в итоге заняло в сумме более нескольких тысяч эпох. Обучение выполнялось на нескольких наборах данных, среди которых: RSCD, CARLA based datasets, CityScapes, TuSimple и другие. Итоговая точность базовой версии алгоритма (В0) по показателю IoU равна 0.6478.

Модель | mIoU, %
------------- | ------------- 
IkshanaNet-3 | 42.07
SegNet | 57.00
ENet | 58.30
ESPNet | 60.30
DeepLab | 63.10
**EfficientUDet** | **64.78**
Fast-SCNN | 68.00
MobileNet V3-Large 1.0 | 72.60
PSPNet | 78.40
HRNetV2 | 81.60
EfficientPS | 84.21
InternImage-H | 86.10

По результатам сравнительного анализа показателей качества разработанной архитектуры EfficietnUDet с показателями качества существующих моделей семантической сегментации было выявлено, что разработанный алгоритм не является самым точным среди ныне существующих. Стоит отметить, что обучение алгоритма выполнялось при относительно слабом аппаратном решении, что все равно не помешало ему достигнуть соизмеримых результатов и опередить несколько ранее предложенных моделей. Результаты исследования в значительной степени отражают потенциал развития данной модели. Дальнейшие исследования позволят повысить финальный показатель качества работы модели

https://elibrary.ru/download/elibrary_53956713_85803964.pdf
