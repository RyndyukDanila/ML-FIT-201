# Регрессия (regression)

### Источник (ссылка).
>[Лесные пожары](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)

### Краткое описание: о чём они, как получены, связанные публикации (если есть).
>This study will consider forest fire data from the Montesinho natural park, from northeast region of Portugal. This park contains a high flora and fauna diversity. Inserted within a supra-Mediterranean climate, the average annualtemperature is within the range 8 to 12 °C. The data used in the experiments was collected from January 2000 to December 2003 and it was built using two sources. 
The first database was collected by the inspector that was responsible for the Montesinhofire occurrences. At a daily basis, every time a forest fire occurred, several features were registered, such as the time, date, spatial location within a 9×9 grid, the type of vegetation involved, the six components of the [FWI system](https://www.nwcg.gov/publications/pms437/cffdrs/fire-weather-index-system) and the total burned area. 
The second database was collected by the Braganca Polytechnic Institute, containing several weather observations (e.g. wind speed) that were recorded with a 30 minute period by a meteorological station located in the center¸ of the Montesinho park. 
The two databases were stored in tens of individual spreadsheets, under distinct formats, and a substantial manual effort was performed to integrate them into a single dataset with a total of 517 entries. 
>>В этом исследовании будут учитываться данные о лесных пожарах в природном парке Монтесинью в северо-восточном регионе в Португалии. Этот парк отличается высоким разнообразием флоры и фауны. В пределах надсредиземноморского климата среднегодовая температура находится в диапазоне от 8 до 12 °C. Данные, использованные в экспериментах, были собраны с января 2000 г. по декабрь 2003 г. и построены с использованием двух источников. 
Первая база данных была собрана инспектором, ответственным за происшествия в Монтезинхофаре. Ежедневно, каждый раз, когда возникал лесной пожар, были зарегистрированы, такие как время, дата, пространственное положение в сетке 9×9, тип вовлеченной растительности, шесть компонентов [системы FWI](https://www.nwcg.gov/publications/pms437/cffdrs/fire-weather-index-system) и общая выгоревшая площадь. 
Вторая база данных была собрана Политехническим институтом Браганца и содержала несколько наблюдений за погодой (например, скорость ветра), которые были записаны с 30-минутным периодом метеорологической станцией, расположенной в центре парка Монтесиньо. 
Две базы данных хранились в десятках отдельных электронных таблиц в разных форматах, и для их интеграции в единый набор данных, содержащий в общей сложности 517 записей, были предприняты значительные усилия вручную.
>
>Relevant Papers:
[Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: [Web-Link](http://www3.dsi.uminho.pt/pcortez/fires.pdf)

### Суть задачи, целевой признак
>This is a difficult regression task, where the aim is to predict the burned area of forest fires, in the northeast region of Portugal, by using meteorological and other data
>>Это сложная регрессионная задача, целью которой является прогнозирование площади лесных пожаров в северо-восточном регионе Португалии с использованием метеорологических и других данных.

>Целевой признак — сгоревшая площадь леса в гектарах (1га = 10 000 м²)

### Признаки объектов, описание этих признаков, тип.
1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9

	>Пространственное расположение в сетке 9×9 природного парка Монтесинью (природный заповедник в Португалии); координаты

3. month - month of the year: "jan" to "dec" 

	>Месяц года, с января по декабрь; порядковый признак

4. day - day of the week: "mon" to "sun"

	>День недели, с понедельника по воскресенье; порядковый признак

5. FFMC - FFMC index from the FWI system: 18.7 to 96.20

	>Код влажности топлива, FFMC обозначает влажность поверхностной подстилки и влияет на воспламенение и распространение огня; вещественный признак

6. DMC - DMC index from the FWI system: 1.1 to 291.3 

7. DC - DC index from the FWI system: 7.9 to 860.6 

	>DMC и DC представляют собой влажность поверхностных и глубоких органических слоев, которые влияют на интенсивность пожара; вещественный признак

8. ISI - ISI index from the FWI system: 0.0 to 56.10

	>ISI показатель, который коррелирует с распространением скорости огня; вещественный признак

9. temp - temperature in Celsius degrees: 2.2 to 33.30

	>Температура в Цельсиях; вещественный признак

10. RH - relative humidity in %: 15.0 to 100

	>Относительная влажность в процентах; вещественный признак

11. wind - wind speed in km/h: 0.40 to 9.40 

	>Скорость метрах в километрах в час; вещественный признак

12. rain - outside rain in mm/m2 : 0.0 to 6.4 

	>Дождь в миллиметрах на квадратный метр; вещественный признак

13. area - the burned area of the forest (in ha): 0.00 to 1090.84 

	>Целевой признак — сгоревшая площадь леса в гектарах (1га = 10 000 м²); вещественный признак

(this output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform).



# Классификация (classification)

### Источник (ссылка).
>[Музыкальный жанр](https://www.kaggle.com/vicsuperman/prediction-of-music-genre)

### Краткое описание: о чём они, как получены, связанные публикации (если есть).
>The Spotify Web API provides artist, album, and track data, as well as audio features and analysis.
>>Веб-API Spotify предоставляет данные об исполнителях, альбомах и дорожках, а также аудио особенности и анализ.

### Суть задачи, целевой признак
>A classification task, where the aim is to predict the music genre using Spotify's wide library of tracks and their audio features.
>>Задача классификации, целью которой является прогнозирование музыкального жанра с использованием широкой библиотеки треков Spotify и их аудио особенностей.

>Целевой признак — музыкальный жанр

### Признаки объектов, описание этих признаков, тип.
1. instance_id - unique ID for each music

	>Уникальный ID для каждого трека; вещественный признак

2. artist_name - artist name

	>Имя артиста; текст

3. track_name - track name

	>Название трека; текст

4. popularity - how popular of this music

	>Популярность, насколько популярна музыка; вещественный признак

5. acousticness - acousticness

	>Акустичность - чем выше значение, тем акустичнее песня; вещественный признак

6. danceability - danceability

	>Танцевальность, чем выше значение, тем танцевальней музыка; вещественный признак

7. duration_ms - the duration of the music in ms

	>Длительность, в миллисекундах; вещественный признак

8. energy - energy

	>Энергичность - чем выше значение, тем энергичнее песня; вещественный признак

9. instrumentalness - instrumentalness

	>Инструментальность - богатсво аранжировки, чем больше значение, тем больше инструментов в треке; вещественный признак

10. key - music key

	>Тональность - в обозначении тональности указывается тоника (она же — основной тон тонического трезвучия); порядковый признак

11. liveness - liveness

	>Живость - вероятность того, что трек был записан на живую; вещественный признак

12. loudness - loudness

	>Громкость - чем выше значение, тем громче музыка; вещественный признак

13. mode - mode

	>Лад - мажорный или минорный; бинарный признак

14. speechiness - speechiness

	>Речь - определяет наличие произносимых слов в треке, чем выше значение, тем больше слов в треке; вещественный признак

15. tempo - tempo

	>Темп - показатель, определяющий скорость исполнения или воспроизведения композиции. BPM — это количество четвертных нот в минуту; вещественный признак;

16. obtained_date - date

	>Дата - число и месяц; порядковый признак

17. valence - valence

	>Валентность - чем выше значение, тем более позитивное настроение у песни; вещественный признак

18. music_genre - music genre

	>Целевой признак - музыкальный жанр (11 значений); категориальный признак
