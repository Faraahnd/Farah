# Final Task Data Scientist - Project Based Virtual Intership Experience Rakamin Academy X Kalbe Nutritionals

# Background Story
- Kamu adalah seorang Data Scientist di kalbe Nutritionals dan sedang 
  mendapatkan project baru dari tim inventory dan tim marketing.
- Dari rim inventory, kamu diminta untuk dapat membantu memprediksi jumlah 
  (quantity) dari total keseluruhan product Kalbe
   1. Tujuan dari project ini adalah untuk mengetahui perkiraan quantity product 
      yang terjual sehingga tim inventory dapat membuat stock persediaan harian 
      yang cukup.
   2. Prediksi yang dilakukan harus harian.
- Dari tim marketing kamu diminta untuk membuat cluster/segment customer 
  berdasarkan beberapa kriteria.
   1. Tujuan dari project ini adalah untuk membuat segment customer.
   2. Segment customer ini nantinya akan digunakan oleh tim marketing untuk 
      memberikan personalized promotion dan sales treatment.
- Tools yang akan kamu gunakan dalam project ini adalah
   1. Python
   2. Jupyter Notebook
   3. Tableau
   4. Dbeaver
   5. PostgreSQL

# Exploratory Data Analysis di Dbeaver
1. Berapa rata-rata umur customer jika dilihat dari maritial statusnya?
   
   ![sql1a](https://github.com/Faraahnd/Farah/assets/143933284/c1769566-3270-40a0-be6b-08af5f02034f)
   
   ![sql1b](https://github.com/Faraahnd/Farah/assets/143933284/7dd03483-40d2-4e9a-be40-16a21e7f9f42)

2. Berapa rata-rata umur customer jika dilihat dari gendernya?

   ![sql2a](https://github.com/Faraahnd/Farah/assets/143933284/882686c0-062b-435c-b364-720866535f66)

   ![sql2b](https://github.com/Faraahnd/Farah/assets/143933284/8481fcf4-5a79-4a80-8420-cc19e5d31a5e)

3. Tentukan nama store dengan total quantity terbanyak!

   ![sql3a](https://github.com/Faraahnd/Farah/assets/143933284/3b9b34b3-beb0-487d-81f5-cd358fd9952f)

   ![sql3b](https://github.com/Faraahnd/Farah/assets/143933284/d0cdc432-728a-4f42-a457-c45376a603c7)

4. Tentukan nama produk terlarik dengan total amount terbanyak!

   ![sql4a](https://github.com/Faraahnd/Farah/assets/143933284/b7a4736a-0947-4af9-8272-16fb0b782df3)

   ![sql4b](https://github.com/Faraahnd/Farah/assets/143933284/20f0d005-7811-47cc-b822-1e787031991c)

# Membuat Dashboard di Tableau

![Dashboard 1](https://github.com/Faraahnd/Farah/assets/143933284/bdd926e6-137f-4553-bd23-b2ce7196d10d)

# Membuat model prediktif menggunakan regresi (Time Series)
1. Import library

   ![1](https://github.com/Faraahnd/Farah/assets/143933284/060210a1-4dac-4ede-a14a-885762a238e0)

2. Import data

   ![2](https://github.com/Faraahnd/Farah/assets/143933284/83bfd041-6fa9-44d9-8bf5-0d27e92f80a9)

3. Data Preparation dan Cleaning

   ![3](https://github.com/Faraahnd/Farah/assets/143933284/bbfe2aec-ee3f-44b4-b29a-59b56531c024)

   ![4](https://github.com/Faraahnd/Farah/assets/143933284/788d61c5-e310-4fb9-82ac-967a1b067613)

   ![5](https://github.com/Faraahnd/Farah/assets/143933284/26a89038-4e64-4c18-9153-5432f9e87672)

   ![6](https://github.com/Faraahnd/Farah/assets/143933284/711adb1e-42d2-4523-b990-996174d1b21b)

   ![7](https://github.com/Faraahnd/Farah/assets/143933284/5417eec1-b788-4bfc-8c17-4cc18935679a)

   ![8](https://github.com/Faraahnd/Farah/assets/143933284/eec03f9b-2423-466d-a764-bd85ec711690)

4. Menggabungkan semua data menjadi 1

   ![9](https://github.com/Faraahnd/Farah/assets/143933284/7352e571-eb24-4578-acba-a9a4fd4e902d)

5. Membuat model Time Series

   ![10](https://github.com/Faraahnd/Farah/assets/143933284/49e93197-3017-497c-bd7f-d99cdebf914f)

   ![11](https://github.com/Faraahnd/Farah/assets/143933284/81a5f621-efc3-40b0-9159-03070a0c9b5f)

6. Memisahkan data

   ![12](https://github.com/Faraahnd/Farah/assets/143933284/b4a6a09f-117b-4203-8957-bb50629c7a0f)

   ![13](https://github.com/Faraahnd/Farah/assets/143933284/21143ed4-8137-40a3-888d-87c749845d04)

7. Model ARIMA

   ![14](https://github.com/Faraahnd/Farah/assets/143933284/c8387ff6-83f9-4fff-9de2-e13801260184)

   ![15](https://github.com/Faraahnd/Farah/assets/143933284/d6f0ccb6-edea-4c5c-bf60-c36449b8608c)

8. Prediksi

   ![16](https://github.com/Faraahnd/Farah/assets/143933284/bb9697e5-8863-4498-96c0-0bcc42da0f7c)

# Membuat Clustering
1. Menggabungkan data

   ![17](https://github.com/Faraahnd/Farah/assets/143933284/bfff4a49-dba9-490b-aed9-0d653ef9b8db)

2. Membuat Clustering KMeans

   ![18](https://github.com/Faraahnd/Farah/assets/143933284/88075c1f-c987-4716-8cba-3a1fd59c495f)

3. Membuat Plot

   ![19](https://github.com/Faraahnd/Farah/assets/143933284/43a9ebed-0e84-4ce8-a605-6ac5fe95c992)

4. WCSS (Within-Cluster Sum of Squares)

   ![20](https://github.com/Faraahnd/Farah/assets/143933284/2ed63a89-2bcc-4f28-b4bb-cbe6bfce9001)

5. Elbow
   
   ![21a](https://github.com/Faraahnd/Farah/assets/143933284/155ee1b6-e883-4814-a433-0fbfd6290dca)
   
   ![21b](https://github.com/Faraahnd/Farah/assets/143933284/33f3d3d4-0494-417a-b82e-c246d8c01107)

6. Model Clustering dengan K yang optimal

   ![22](https://github.com/Faraahnd/Farah/assets/143933284/1c05b98c-e137-45dd-82aa-cb501195faa5)

   







   








