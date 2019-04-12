### Business and Data Understanding

As talk on Airbnb kaggle data website, the following Airbnb activity is included in this Boston dataset: 

* Calendar, including listing id and the price and availability for that day
* Listings, including full descriptions and average review score 
* Reviews, including unique id for each reviewer and detailed comments 


Let us take a look on these three csv files.

#### Calendar


It shows that the hosts are not avaible everyday and price may be changed at the busiest seasons. 

* What is the most expensive season in Boston? 
* Which hosts are the most favorite？

#### Listings


Summary information on listing in Boston.It contains location, host information, cleaning and guest fees, amenities and so on.
We may find some import factors on price.

* What are the top factors strong relation to price?
* How to predict price？


#### Reviews


We can find many interesting opinions,sush as 

* What are the most attractive facilities? It is big bed, large room or location?
* What will lead to bad impression？

### Data preparing

#### Clean Calendar



![png](https://raw.githubusercontent.com/ahomer/ahomer.github.io/master/data/assets/images/output_16_1.png)


* So we can see the most expensive season is from August to November，especial September and October. 
* You can get a lowest price if you go to Boston at February.



* The most expensive listing_id is 447826.Go to Boston and experience one night.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>301</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>447826</td>
    </tr>
    <tr>
      <th>listing_url</th>
      <td>https://www.airbnb.com/rooms/447826</td>
    </tr>
    <tr>
      <th>scrape_id</th>
      <td>20160906204935</td>
    </tr>
    <tr>
      <th>host_url</th>
      <td>https://www.airbnb.com/users/show/2053557</td>
    </tr>
    <tr>
      <th>name</th>
      <td>Sweet Little House in JP, Boston</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>1</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>2</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1</td>
    </tr>
    <tr>
      <th>amenities</th>
      <td>{TV,"Cable TV",Internet,"Wireless Internet",Ki...</td>
    </tr>
  </tbody>
</table>
</div>



![Sweet Little House in JP, Boston](https://raw.githubusercontent.com/ahomer/ahomer.github.io/master/data/assets/images/447826.png)

#### Clean Listings


Let us calculate the mean/std of 'Price'.

* Assuming that prices obey normal distribution
* The price should be between mean-2*std~mean+2*std



![png](https://raw.githubusercontent.com/ahomer/ahomer.github.io/master/data/assets/images/output_27_1.png)



#### Clean Reviews

Review the reviews.csv file,you will find there are different languages.We just need to keep the english comment.<br>
We need a lib 'langdetect'.






### Modeling and  evaluation

Let's try to predict the price based on the columns in the listing we selected.


* What are the top factors strong relation to price?




![png](https://raw.githubusercontent.com/ahomer/ahomer.github.io/master/data/assets/images/output_39_1.png)


Top 6 factors strong relation to price:

* bedrooms
* room type : Private room
* number of reviews
* accommodates
* bathrooms
* review scores rating


### Deployment

Mostly,the model will be deplyed on product environment based on a RPC server or http server.<br>
You can deploy the model with Tornado(python web framework).


