# Travel_Itinerary_Data
This database contains four folders: 1. Attractions, 2. Tourists, 3. Travelogues, 4. Benchmark_Model_Code.py.
1. Attractions 

(1) The attraction_informations table contains basic information data (Attraction name, Rating of attraction, Ticket price, Visiting time) for 208 attractions in Beijing.

(2) The traffic_time_cost table contains traffic data ([Traffic time to other attractions,Traffic cost to other attractions]) for 208 attractions in Beijing.

(3) The online_review_texts, online_review_texts1, online_review_texts2 folders store online review texts for the 208 attractions in Beijing.

2. Tourists

This folder contains online travelogue texts published by 100 target tourists.

3. Travelogues

(1) The raw_data table contains basic information (Travelogue_title, Views, Likes, Comments, Travel_time, Visiting season, Travel companions, Per capita consumption, Online travelogue text) for 1000 popular travelogues in Beijing.

(2) The data table contains basic information (Travelogue_title, Views, Likes, Comments, Travel_time, Visiting season, Travel companions, Per capita consumption, Route) for the 1000 popular travelogues in Beijing.

(3) Based on the data table, 100 travelogues are randomly selected as a test set and stored in the test table, while the basic information of the other 900 travelogues were stored in the train table.

4. Benchmark_Model_Code.py

This file contains the specific implementation process of five benchmark models.

(1) Random : For each target tourist, this method randomly selects a scenic spot as the starting point and randomly generates the next unanswered scenic spot. Under the constraint that the total daily time is less than 10 hours and the total multi day cost is less than the expected consumption of the target tourist, it recommends a multi day route that meets the tourist's travel time.

(2) POP : This method determines the popularity of each attraction based on its strategy number. For each target tourist, the method model selects the most popular scenic spot as the starting point, and selects the second most popular scenic spot as the next scenic spot. Under the constraint that the total daily time is less than 10 hours and the total multi day cost is less than the target tourist's expected consumption, the model recommends a multi day route that meets the tourist's travel time.

(3) NSM : For each target tourist, this method randomly selects a scenic spot as the starting point and counts the transportation time from that scenic spot to other scenic spots. The shortest scenic spot is selected as the next scenic spot, and under the constraint of a total daily time of less than 10 hours and a total multi day cost less than the target tourist's expected consumption, a multi day route that meets their travel time is recommended for tourists.

(4) Item CF : Assuming that the first attraction on the actual route of the target tourist is their favorite attraction. Item CF calculates the similarity between other attractions in the attraction database and favorite attractions. For each target tourist, the Item CF model selects the most similar attraction as the starting point, and selects the second most similar attraction as the next attraction. Under the constraint that the total daily time is less than 10 hours and the total multi day cost is less than the target tourist's expected consumption, the Item CF model recommends a multi day route that meets the tourist's travel time.

For any queries please email Xiangqian Li at 2918643257@qq.com
