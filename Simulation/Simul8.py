import pythoncom
#import numpy as np
from win32com import client
import traceback

class SimEvents:
    """
    This class contains special Simul8 events that are intercepted within the loop.

    ...

    Methods
    -------
    set_master(master)
        Instantiates an instance of the Simul8API to access it within the events
    OnS8SimulationOpened()
        Is triggered as soon as the simulation is opened
    OnS8SimulationEndRun()
        Is triggered as soon as a simulation run is completed and sets and sets the variable that that controls the
        query loop that intercepts Simul8 events
    OnS8SimulationCustomEvent(event)
        Is triggered as soon as a custom event is called in a Visual Logic function and processes the commands
        resulting from the event
    """

    def set_master(self, master):
        """
        Instantiates an instance of the Simul8API to access it within the events

        Parameters
        ----------
            master: Simul8API
                 An instance of the Simul8API

        """
        self.master = master

    def OnS8SimulationOpened(self):
        """
        Is triggered as soon as the simulation is opened

        """
        print("Simulation Opened")

    def OnS8SimulationEndRun(self):
        """
        Is triggered as soon as a simulation run is completed and sets and sets the variable that that controls the
        query loop that intercepts Simul8 events

        """
        # Instantiate global variable to use it outside of function
        self.master.listenForMessages = False

    def OnS8SimulationCustomEvent(self, event):
        """
        Is triggered as soon as a custom event is called in a Visual Logic function and processes the commands
        resulting from the event

        Parameters
        ----------
            event: str
                 A text variable containing commands to be processed by the simulation

        """
        requests = []
        messages = event.split(";")
        t_step = round(float(messages.pop(0)), 3)

        for message in messages:
            stage_1 = message.split(":")
            command = stage_1[0]
            stage_2 = stage_1[1]
            attributes = []
            for attribute in stage_2.split(","):
                attributes.append(attribute)
            if command == "ProcessingTimes":
                processing_times = {}
                for attribute in attributes:
                    processing_time = attribute.split(".")
                    machine = processing_time[0]
                    time = processing_time[1]
                    processing_times[machine] = int(time)
                self.master.processing_times = processing_times
                self.master.t_step = t_step
            else:
                request = {"Command": command, "Attributes": attributes}
                requests.append(request)
                self.master.requests = requests
                self.master.t_step = t_step
                self.master.new_request = True


class Simul8API:
    """
    This class represents an interface between Python and the simulation software Simul8. The class is used to directly
    address Simul8 objects or trigger Visual Logic functions.

    ...

    Attributes
    ----------
    S8 : None
        A Simul8 instance
    environment: Environment
        An instance of the Environment class
    listenForMessages: None
        A variable that controls the query loop that intercepts Simul8 events
    requests: None
        A variable to which incoming commands of the simulation are attached
    t_step: float
        A variable to which the current simulation time is attached
    new_request: False
         A variable that indicates whether a new request with commands is available
    processing_times: None
        A variable to which the process times set for the activities of the simulation model are attached at the
        start of the simulation

    Methods
    -------
    open(simul8_project_path)
        Starts the event handler and opens the simulation file passed as an attribute
    set_sim_object()
        Instantiates the connections to the objects of the simulation model
    run(end_time, sim_speed):
        Starts a simulation run at the simulation speed specified as an attribute
    reset()
        Resets the simulation
    link_sim_objects(source)
        Connects the simulation object passed as an attribute (picking system) with the colour printing system
    wi_to_queue(nb_wi, queue)
        Generates new work items within a Simul8 queue object passed as an attribute
    close()
        Closes the simulation software
    get_request()
        Returns the requests variable to which the commands of the simulation are assigned
    sim_to_excel()
        Exports the simulation spreadsheets linked to Excel files and converts them into Excel files
    excel_to_sim()
        Imports the Excel files linked to the simulation and converts them into spreadsheets
    get_processing_times()
        Exports the process times stored in the activities of the simulation model
    """

    def __init__(self, environment):
        self.S8 = None
        self.environment = environment

        self.listenForMessages = None
        self.requests = None
        # Windows Startzeit: 45547.375 = 12.09.24 09:00:00 Uhr
        self.t_step = 45547.000
        self.new_request = False
        self.processing_times = None

    def open(self, simul8_project_path):
        """
        Starts the event handler and opens the simulation file passed as an attribute

        Parameters
        ----------
            simul8_project_path: str
                 Path to the simulation file which is to be opened

        """
        # Initialize COM libraries
        pythoncom.CoInitialize()
        # Request Simul8 object from system
        self.S8 = client.Dispatch("Simul8.S8Simulation")
        # Register Event Handler
        event_handler = client.WithEvents(self.S8, SimEvents)
        event_handler.set_master(master=self)
        # Open Simul8 Project
        try:
            self.S8.Open(simul8_project_path)
            self.set_sim_objects()
            self.get_processing_times()
        except Exception:
            print(traceback.format_exc())

    def set_sim_objects(self):
        """
        Instantiates the connections to the objects of the simulation model

        """
        self.new_order_16 = self.S8.SimObject("NewOrder_16")
        self.new_order_6 = self.S8.SimObject("NewOrder_6")
        self.new_order_4 = self.S8.SimObject("NewOrder_4")
        self.new_order_25 = self.S8.SimObject("NewOrder_25")
        self.new_order_15 = self.S8.SimObject("NewOrder_15")
        self.new_order_fd = self.S8.SimObject("NewOrder_FD")
        self.move_on_16 = self.S8.SimObject("MoveOn_16")
        self.move_on_6 = self.S8.SimObject("MoveOn_6")
        self.move_on_4 = self.S8.SimObject("MoveOn_4")
        self.move_on_25 = self.S8.SimObject("MoveOn_25")
        self.move_on_15 = self.S8.SimObject("MoveOn_15")

    def run(self, end_time: float, sim_speed: float=100):
        """
        Starts a simulation run at the simulation speed specified as an attribute

        Parameters
        ----------
            end_time: float
                End time of the simulation
            sim_speed: float
                The simulation speed in per cent

        """
        self.S8.RunSim(end_time)
        self.S8.SimSpeed = sim_speed

    def reset(self):
        """
        Resets the simulation

        """
        self.t_step = 45547.000
        self.S8.ResetSim(0)

    def link_sim_objects(self, source):
        """
        Connects the simulation object passed as an attribute (picking system) with the colour printing system

        Parameters
        ----------
            source: float
                An abbreviation of the simulation object that is to be connected to the colour printing system

        """
        if source == "16":
            self.S8.LinkSimObjects(self.move_on_16, self.new_order_fd, 0)
        elif source == "6":
            self.S8.LinkSimObjects(self.move_on_6, self.new_order_fd, 0)
        elif source == "4":
            self.S8.LinkSimObjects(self.move_on_4, self.new_order_fd, 0)
        elif source == "25":
            self.S8.LinkSimObjects(self.move_on_25, self.new_order_fd, 0)
        elif source == "15":
            self.S8.LinkSimObjects(self.move_on_15, self.new_order_fd, 0)

    def wi_to_queue(self, nb_wi, queue=None):
        """
        Generates new work items within a Simul8 queue object passed as an attribute

        Parameters
        ----------
            nb_wi: int
                The number of work items to generate
            queue: str
                An abbreviation of the queue to which the work items are to be added

        """
        def check_wis_in_queue():
            wis_in_queue = int(queue.CountContents)
            if wis_in_queue != nb_wi:
                print("Less WIs in Queue than it should be")
                dif = nb_wi - wis_in_queue
                for _ in range(dif):
                    queue.AddWI(0, "Main Work Item Type")
                check_wis_in_queue()

        if queue == "16":
            queue = self.new_order_16
        elif queue == "6":
            queue = self.new_order_6
        elif queue == "4":
            queue = self.new_order_4
        elif queue =="25":
            queue = self.new_order_25
        elif queue =="15":
            queue = self.new_order_15
        for _ in range(nb_wi):
            queue.AddWI(0, "Main Work Item Type")

    def close(self):
        """
        Closes the simulation software

        """
        self.S8.Close()
        # Uninitialize COM libraries to avoid memory leak
        pythoncom.CoUninitialize()

    def get_request(self):
        """
        Returns the requests variable to which the commands of the simulation are assigned

        :return:
            The requests variable to which the commands of the simulation are assigned

        """
        return self.requests

    def sim_to_excel(self):
        """
        Exports the simulation spreadsheets linked to Excel files and converts them into Excel files

        """
        self.S8.CallVL("Export Sim Data")

    def excel_to_sim(self):
        """
        Imports the Excel files linked to the simulation and converts them into spreadsheets

        """
        self.S8.CallVL("Import Sim Data")

    def get_processing_times(self):
        """
        Exports the process times stored in the activities of the simulation model

        """

        self.S8.CallVL("Get Processing Times")


if __name__ == "__main__":
    sim = Simul8API(None)
    sim.open("JobShop.S8")
