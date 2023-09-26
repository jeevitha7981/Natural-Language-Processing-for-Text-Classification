#!/usr/bin/env python
# coding: utf-8

# In[ ]:


order = pd.read_csv("D:\List of Orders.csv")
order.head()


# In[ ]:


details = pd.read_csv("D:\Order Details.csv")
details.head()


# In[ ]:


target = pd.read_csv("D:\Sales target.csv")
target.head()


# In[ ]:


# Cleaning the order dataset
order.info()


# In[ ]:


# Changing the Order Date variable to datetime data type
order['Order Date'] = order['Order Date'].astype('datetime64[ns]')


# In[ ]:


# Null values
details.isnull().sum()


# In[ ]:


# Droping Null Values
order = order.dropna()
order.info()


# In[ ]:


# Cleaning the detail dataset
details.info()


# In[ ]:


# Null values
details.isnull().sum()


# In[ ]:


# Chaning the Category and Sub-category variable to categorical data type
details['Category'] = details['Category'].astype('category')
details['Sub-Category'] = details['Sub-Category'].astype('category')
details.info()


# In[ ]:


# Cleaning Target dataset
target.info()


# In[ ]:


# Coverting Category variable to category data
target['Category'] = target['Category'].astype('category')
target.info()


# In[ ]:


# Cleanded Details data
details.head()


# In[ ]:


# Cleaned Order Data
order.head()


# In[ ]:


# Cleaned Target Dataset
target.head()


# In[ ]:


profits = details.groupby('Order ID').sum().reset_index()
profits.head()


# In[ ]:


df = pd.merge(order, profits)
df.head()


# In[ ]:


import calendar

df['Year'] = pd.DatetimeIndex(df['Order Date']).year
df['Month_Number'] = pd.DatetimeIndex(df['Order Date']).month
df['Month'] = df['Month_Number'].apply(lambda x: calendar.month_abbr[x])

year_month = df.groupby(['Year', 'Month', 'Month_Number']).sum().sort_values(['Year', 'Month_Number'])
year_month


# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px

# Assuming you have already defined the 'year_month' DataFrame
year_month = year_month.reset_index()
year_month["Color"] = np.where(year_month["Profit"] < 0, 'Loss', 'Profit')
year_month_2018 = year_month[year_month['Year'] == 2018]

fig = px.bar(year_month_2018, x='Month_Number', y='Profit', color='Color',
             title="Monthly Profit in 2018",
             labels=dict(Month_Number="Month", Profit="Profit", Color="Results"),
             color_discrete_map={
                 'Loss': '#EC2049',
                 'Profit': '#2F9599'},
             hover_data=["Month", "Profit"],
             template='plotly_white')

fig.update_layout(yaxis_tickprefix='₹', yaxis_tickformat=',.2f')

fig.update_layout(
    xaxis=dict(
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
)

fig.show()


# In[ ]:


year_month_2019 = year_month[year_month['Year']==2019]
fig = px.bar(year_month_2019, x='Month_Number', y='Profit', color='Color',
             title="Monthly Profit in 2019",
             labels=dict(Month_Number="Month", Profit="Profit", Color="Results"),
             color_discrete_map={
                 'Loss': '#EC2049',
                 'Profit': '#2F9599'},
             hover_data=["Month", "Profit"],
             template='plotly_white')

fig.update_layout(yaxis_tickprefix = '₹', yaxis_tickformat = ',.2f')

fig.update_layout(
    xaxis = dict(
        tickvals = [1, 2, 3, 4, 5, 6, 7,8 ,9, 10, 11, 12],
        ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
)
fig.show()


# In[ ]:


orders_by_state = order.groupby(['State']).size().reset_index(name='Total Orders').sort_values(['Total Orders'])
orders_by_state


# In[ ]:


profit_by_state = df.groupby('State').sum().reset_index().sort_values(['Profit'])
profit_by_state["Color"] = np.where(profit_by_state["Profit"]<0, 'Loss', 'Profit')


fig = px.bar(profit_by_state, x='State', y='Profit',
             color='Color', color_discrete_map={
                 'Loss': '#EC2049',
                 'Profit': '#2F9599'},
             title="Profit by State",
             labels=dict(Color="Results"),
             template='plotly_white')

# Disabling Zoom
fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

fig.update_layout(yaxis_tickprefix = '₹', yaxis_tickformat = ',.2f')

fig.update_xaxes(
        tickangle = -90,
        title_text = "States",
)


fig.show()


# In[ ]:


import plotly.graph_objects as go

top_customers = df.groupby('CustomerName').sum().reset_index().sort_values(['Quantity'], ascending=False).head(5)

colors = ['lightslategray',] * 5
colors[0] = 'crimson'

fig = go.Figure(data=[go.Bar(
    x=top_customers['CustomerName'],
    y=top_customers['Quantity'],
    marker_color=colors # marker color can be a single color value or an iterable,
)])

fig.update_layout(title_text='Top 5 Customers', template='plotly_white')
fig.update_xaxes(title_text='Customers')
fig.update_yaxes(title_text='Total Orders')

fig.show()


# In[ ]:


details_category = details.groupby('Category').sum().reset_index()
fig = px.pie(details_category, values='Quantity', names='Category', color='Category',
             color_discrete_map={'Clothing':'cyan',
                                 'Electronics':'royalblue',
                                 'Furniture':'darkblue'},
            title='Total Quantity Sold per Category')
fig.show()


# In[ ]:


details_subcategory = details.groupby('Sub-Category').sum().reset_index()
fig = px.pie(details_subcategory, values='Quantity', names='Sub-Category', color='Sub-Category',
            title='Total Quantity Sold per Sub-Category')
fig.show()


# In[ ]:


date_orders = order.groupby('Order Date').size().reset_index(name="Orders")
date_orders['Month'] = pd.DatetimeIndex(date_orders['Order Date']).month
date_orders['Year'] = pd.DatetimeIndex(date_orders['Order Date']).year

date_orders_2018 = date_orders[date_orders['Year']==2018]
date_orders_2019 = date_orders[date_orders['Year']==2019]

month_2018 = date_orders_2018.groupby('Month').sum().reset_index()
month_2019 = date_orders_2019.groupby('Month').sum().reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    name='2018',
    x=month_2018['Month'],
    y=month_2018['Orders'],
    connectgaps=True # override default to connect the gaps
))
fig.add_trace(go.Scatter(
    name='2019',
    x=month_2019['Month'],
    y=month_2019['Orders'],
    connectgaps=True # override default to connect the gaps
))
fig.update_layout(title_text='Monthly Quantity Sold',
                 template='plotly_dark')
fig.update_xaxes(title_text='Time')
fig.update_yaxes(title_text='Orders')
fig.update_layout(
    xaxis = dict(
        tickvals = [1, 2, 3, 4, 5, 6, 7,8 ,9, 10, 11, 12],
        ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
)

fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

fig.show()


# In[ ]:


fig = px.bar(orders_by_state, y='State', x='Total Orders',
             title="Total Orders by State", 
             color_discrete_sequence=["springgreen"],
             template='plotly_white')

# Disabling Zoom
fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

fig.show()


# In[ ]:


#
orders_by_city = order.groupby(['City']).size().reset_index(name='Total Orders').sort_values(['Total Orders'])

fig = px.bar(orders_by_city, y='City', x='Total Orders',
             title="Total Orders by City",
             template='simple_white')

fig.layout.yaxis.tickmode='linear'
# Disabling Zoom
fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

fig.show()


# In[ ]:


target_category = target.groupby('Category').max().reset_index()
details_category = details.groupby('Category').sum().reset_index()

target_category['Actual_Amount'] = details_category['Profit']

fig = go.Figure(data=[
    go.Bar(name='Target', x=target_category['Category'], y=target_category['Target'],
          marker_color='#2b2d42'),
    go.Bar(name='Actual Amount', x=target_category['Category'], y=target_category['Actual_Amount'],
          marker_color='#d90429')
])

fig.update_layout(title_text='Actual vs Target Sales',
                 template='plotly_white')

fig.update_xaxes(title_text='Categories')
fig.update_yaxes(title_text='Amount')

fig.update_layout(yaxis_tickprefix = '₹', yaxis_tickformat = ',.2f')


fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

fig.show()


# In[ ]:


customer_seg = df.groupby('CustomerName').sum().reset_index()
customer_seg = customer_seg[['CustomerName', 'Amount', 'Quantity']]
customer_seg.head()


# In[ ]:


get_ipython().system('pip uninstall threadpoolctl')
get_ipython().system('pip install threadpoolctl')

# Standardizing
customer_seg2 = customer_seg[['Amount', 'Quantity']]
scaler = StandardScaler()
scaler.fit(customer_seg2)

customers_normalized = scaler.transform(customer_seg2)
customers_normalized
# Elbow Method to find best number of clusters
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customers_normalized)
    sse[k] = kmeans.inertia_  # Assign SSE to closest cluster centroid

# Plotting SSE
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(sse.keys()),
    y=list(sse.values()),
    connectgaps=True  # Override default to connect the gaps
))

fig.update_layout(
    title_text='The Elbow Method',
    template='plotly_white'
)
fig.update_xaxes(title_text='k')
fig.update_yaxes(title_text='SSE')
fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True
fig.show()


# In[ ]:


# KMeans
model = KMeans(n_clusters=3)
model.fit(customers_normalized)
customer_seg['Cluster'] = model.labels_ + 1
customer_seg['Cluster'] = customer_seg['Cluster'].astype('category')
customer_seg.head()


# In[ ]:


customer_seg.groupby('Cluster').agg({
    'Amount':'mean',
    'Quantity':'count'}).round(2)


# In[ ]:


fig = px.scatter(customer_seg, x="Quantity", y="Amount",
                 color="Cluster",
                 template='plotly_white',
                 title="Amount vs Quantity - Customer Segmentation")
fig.layout.xaxis.fixedrange = True
fig.layout.yaxis.fixedrange = True

fig.show()


# In[ ]:
