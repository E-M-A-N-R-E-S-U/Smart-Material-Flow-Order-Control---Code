import random
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


class RandomOrder:
    """
    A class for generating random orders

    ...

    Attributes
    ----------
    number : int
        The order number of the order
    quantity : int
        The number of products included in the order
    cross_section : int
        The cross-section of the products included in the order
    start_disk: bool
        Feature on whether a start disk is installed on the products
    center_disk: int
        Feature on whether a center disk is installed on the products and if so, how much
    center_flange: int
        Feature on whether a center flange is installed on the products and if so, how much
    end_disk: bool
        Feature on whether a end disk is installed on the products
    end_hybrid_disk: bool
        Feature on whether a end hybrid disk is installed on the products
    b_w_print: bool
        Feature on whether the products have to be black and white printed and if so, how much
    color_print: bool
        Feature on whether the products have to be color printed and if so, how much
    deadline: float
        Deadline of the order


    Methods
    -------
    get_data()
        Returns a dictionary containing the features describing the order
    """

    def __init__(self, number=None, quantity=None, cross_section=None):
        self.number = random.randrange(10000, 99999) if not number else number
        self.quantity = random.randrange(50, 100) if not quantity else quantity
        self.cross_section = random.choice([15, 25, 4, 6, 16]) if not cross_section else cross_section
        self.start_disk = 1

        self.center_disk = random.randint(0,1)
        if self.center_disk:
            self.center_disk = random.randrange(1,10)
            self.center_flange = 0
        else:
            self.center_flange = random.randrange(1,10) # random.choice([abs(self.center_disk-1), 0])

        self.end_disk = random.randint(0,1)
        self.end_hybrid_disk = abs(self.end_disk-1)

        self.connector = random.randrange(0,self.center_disk if self.center_disk else self.center_flange)

        self.b_w_print = random.randint(0,1)
        if self.b_w_print:
            self.b_w_print = random.randrange(1,self.start_disk+self.center_disk+self.center_flange+self.end_disk+self.end_hybrid_disk)
            self.color_print = 0
        else:
            self.color_print = random.randrange(1,self.start_disk+self.center_disk+self.center_flange+self.end_disk+self.end_hybrid_disk)
        self.deadline = 3.0

    def get_data(self) -> dict:
        """
        Returns a dictionary containing the features describing the order.

        :return:
            A dictionary containing the features describing the order.

        """
        new_entry = {"OrderNumber": self.number,
                     "n_Items": self.quantity,
                     "ItemType": self.cross_section,
                     "Deadline": self.deadline,
                     "S": self.start_disk,
                     "M": self.center_disk,
                     "MF": self.center_flange,
                     "E": self.end_disk,
                     "EH": self.end_hybrid_disk,
                     "VB": self.connector,
                     "SWD": self.b_w_print,
                     "FD": self.color_print,
                     "NewEntry": "True"}

        return new_entry


class OrderIntake:
    """
    A class for managing incoming orders.

    ...

    Attributes
    ----------
    columns : list
        A list containing the column names of the order intake data frame
    df : int
        A data frame containing all incoming orders

    Methods
    -------
    len()
        Returns the length of the order intake data frame
    duplicat(order_number)
        Checks whether an order number is already contained in the order intake
    get()
        Returns the order intake data frame
    get_order_by_order_number(order_number)
        Searches for an order in the order intake dataframe by its order number and returns the order if it was found
    get_orders_of_type(item_type)
        Searches for all orders of a given item type and returns them afterward
    get_new_orders_of_type()
        Searches for new orders of a given item type and returns them afterward
    get_order_by_index(idx)
        Returns an order that is located at the position of a given index in the DataFrame
    append(data)
        Appends a passed order to the order intake data frame
    get_random_order()
        Returns a random order from the order intake data frame
    reset_incoming_orders()
        Sets the information on whether an order has been newly received to true for all orders
    clear()
        Clears the order intake data frame
    to_csv(directory)
        Converts the order intake data frame into a csv file
    to_excel(directory)
        Converts the order intake data frame into a Excel file
    """

    def __init__(self):
        self.columns = ["OrderNumber", "n_Items", "ItemType", "Deadline", "S", "M", "MF", "E", "EH", "VB", "SWD", "FD", "NewEntry"]
        self.df = pd.DataFrame(columns=self.columns)

        # file_directory = os.path.dirname(__file__)
        # self.excel_directory = os.path.join(file_directory, "OrderIntake.xlsx")
        # self.to_excel()

    def __len__(self) -> int:
        """
        Returns the length of the order intake data frame

        :return:
            Length of the order intake data frame as integer

        """
        return len(self.df)

    def duplicat(self, order_number):
        """
        Checks whether an order number is already contained in the order intake

        Parameters
        ----------
            order_number: int
                The order number for which you want to check whether it already exists

        :return:
            Information about whether the order number already exists as bool

        """
        series = self.df["OrderNumber"]
        if order_number in series.values:
            return True
        else:
            return False

    def get(self):
        """
        Returns the order intake data frame

        :return:
            Order intake data frame

        """
        return self.df

    def get_order_by_order_number(self, order_number) -> pd.DataFrame:
        """
        Searches for an order in the order intake dataframe by its order number and returns the order if it was found

        Parameters
        ----------
            order_number: int
                The order number of the order you want to get

        :return:
            The order you want to get

        """
        df = self.df.loc[self.df["OrderNumber"] == order_number]
        return df

    def get_orders_of_type(self, item_type) -> pd.DataFrame:
        """
        Searches for all orders of a given item type and returns them afterward

        Parameters
        ----------
            item_type: int
                The item type of the order you want to get

        :return:
            The orders you want to get

        """
        df = self.df.loc[self.df["ItemType"] == int(item_type)]
        return df

    def get_new_orders_of_type(self, item_type) -> pd.DataFrame:
        """
        Searches for new orders of a given item type and returns them afterward

        Parameters
        ----------
            item_type: int
                The item type of the order you want to get

        :return:
            The new orders you want to get

        """
        df = self.get_orders_of_type(item_type)
        df = df.loc[df["NewEntry"] == "True"]
        order_numbers = df["OrderNumber"].to_list()
        for order_number in order_numbers:
            self.df.loc[self.df["OrderNumber"] == order_number, "NewEntry"] = "False"
        return df

    def get_order_by_index(self, idx: int) -> pd.Series:
        """
        Returns an order that is located at the position of a given index in the DataFrame

        Parameters
        ----------
            idx: int
                The index at which the order you want to get is located in the data frame

        :return:
            The new orders you want to get

        """
        order = self.df.iloc[idx]
        return order

    def append(self, data: dict) -> None:
        """
        Appends a passed order to the order intake data frame

        Parameters
        ----------
            data: int
                The order you want to append to the order intake data frame

        """
        df_of_new_data = pd.DataFrame([data])
        self.df = pd.concat([self.df, df_of_new_data], ignore_index=True)

    def get_random_order(self) -> pd.Series:
        """
         Returns a random order from the order intake data frame

        :return:
            A random order from the order intake data frame

        """
        random_id = random.randrange(1, len(self.df))
        order = self.df.iloc[random_id]
        return order

    def reset_incoming_orders(self):
        """
         Sets the information on whether an order has been newly received to true for all orders

        """
        self.df["NewEntry"] = "True"

    def clear(self) -> None:
        """
        Clears the order intake data frame

        """
        self.df = self.df[0:0]

    def to_csv(self, directory=None) -> None:
        """
        Converts the order intake data frame into a csv file

        Parameters
        ----------
            directory: str
                The directory where the csv file should be saved

        """
        if not directory:
            self.df.to_excel(self.excel_directory, sep=";", index=False)

        self.df.to_csv(directory, sep=";", index=False)

    def to_excel(self, directory=None) -> None:
        """
        Converts the order intake data frame into a Excel file

        Parameters
        ----------
            directory: str
                The directory where the Excel file should be saved

        """
        if not directory:
            self.df.to_excel(self.excel_directory, index=False)

        self.df.to_excel(directory, index=False)
