""" Profile datastore API

Copyright PolyAI Limited
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Dict, List

from numpy.random import RandomState

from nlu.address import AddressParser
from readers import Profile, Slot


def _standardise_postcode(text: str) -> str:
    return text.upper().replace(" ", "").replace("-", "")


class ProfileDatastore(object):
    """ A DB store for user profiles """

    def __init__(
        self,
        scenario_id2profile: Dict[str, Profile],
        searchable_by_postcode: bool = True,
        searchable_by_name: bool = False,
        searchable_by_dob: bool = False,
        searchable_by_email: bool = False,
        searchable_by_floa: bool = False,
        return_all_profiles: bool = False,
        seed=6351
    ):
        """ Initialise

        Args:
            scenario_id2profile: a dictionary [scenario_id, profile]
            searchable_by_postcode: whether the api allows search by postcode
            searchable_by_name: whether the api allows search by name
            searchable_by_dob: whether the api allows search by date of birth
            searchable_by_email: whether the api allows search by email
            searchable_by_floa: whether the api allows search by address
            return_all_profiles: whether queries return all profiles in DB
            seed: random seed
        """
        self._scenario_id2profile = scenario_id2profile
        self.postcode_is_searchable = searchable_by_postcode
        self.name_is_searchable = searchable_by_name
        self.dob_is_searchable = searchable_by_dob
        self.email_is_searchable = searchable_by_email
        self.floa_is_searchable = searchable_by_floa
        self._return_all_profiles = return_all_profiles
        self._rng = RandomState(seed=seed)
        self._address_parser = AddressParser.create("simple")
        if searchable_by_email:
            self.email_hashmap = defaultdict(list)
            for p in self._scenario_id2profile.values():
                self.email_hashmap[p.email].append(p)
        if searchable_by_floa:
            self.num_and_street2profile = defaultdict(list)
            for p in self._scenario_id2profile.values():
                _, num, street = self._address_parser.parse(
                    p.address_first_line
                )[0]
                if not num and not street:
                    continue
                self.num_and_street2profile[
                    (num, street)
                ].append(p)

    def find_by_postcode(self, postcode: str) -> List[Profile]:
        """ Retrieve profiles that match a given postcode """
        if not self.postcode_is_searchable:
            return []
        if self._return_all_profiles:
            return [p for _, p in self._scenario_id2profile.items()]
        retrieved = []
        postcode = _standardise_postcode(postcode)
        for _, p in self._scenario_id2profile.items():
            if postcode == _standardise_postcode(p.postcode):
                retrieved.append(p)
        return retrieved

    def find_by_name(self, name: str) -> List[Profile]:
        """ Retrieve profiles that match a given full name """
        if not self.name_is_searchable:
            return []
        if self._return_all_profiles:
            return [p for _, p in self._scenario_id2profile.items()]
        retrieved = []
        name = name.lower()
        for _, p in self._scenario_id2profile.items():
            if name == p.name_full.lower():
                retrieved.append(p)
        return retrieved

    def find_by_dob(self, dob: dt.date) -> List[Profile]:
        """ Retrieve profiles that match a given date of birth """
        if not self.dob_is_searchable:
            return []
        if self._return_all_profiles:
            return [p for _, p in self._scenario_id2profile.items()]
        dob_str = dob.isoformat()
        retrieved = []
        for _, p in self._scenario_id2profile.items():
            if dob_str == p.dob_str:
                retrieved.append(p)
        return retrieved

    def find_by_email(self, email: str) -> List[Profile]:
        """ Retrieve profiles that match a given email """
        if not self.email_is_searchable:
            return []
        if self._return_all_profiles:
            return [p for _, p in self._scenario_id2profile.items()]
        return self.email_hashmap[email]

    def find_by_floa(self, floa: str) -> List[Profile]:
        """ Retrieve profiles that match a given address """
        if not self.floa_is_searchable:
            return []
        if self._return_all_profiles:
            return [p for _, p in self._scenario_id2profile.items()]
        try:
            _, num, street = self._address_parser.parse(floa)[0]
        except Exception:
            num, street = '', ''
        if not num and not street:
            return []
        return self.num_and_street2profile.get((num, street), [])

    def is_searchable_by(self, item: str):
        """ Whether users can be collected at item turn"""
        if item == Slot.POSTCODE:
            return self.postcode_is_searchable
        elif item == Slot.NAME:
            return self.name_is_searchable
        elif item == Slot.DOB:
            return self.dob_is_searchable
        elif item == Slot.ADDRESS:
            return self.floa_is_searchable
        else:
            raise ValueError(f"Unknown item {item}")

    def get_random(self, n: int) -> List[Profile]:
        """Get up to N random profiles from the DB"""
        all_profiles = [
            p for _, p in sorted(
                self._scenario_id2profile.items(),
                key=lambda x: x[0]
            )
        ]
        indices = self._rng.permutation(len(all_profiles))[:n]
        return [all_profiles[i] for i in indices]

    def find_by_oracle(self, scenario_id: str) -> List[Profile]:
        """ Retrieve profiles with a given scenario_id"""
        try:
            p = self._scenario_id2profile[scenario_id]
        except KeyError:
            return []
        profiles = []
        profiles.extend(self.find_by_postcode(p.postcode))
        profiles.extend(self.find_by_name(p.name_full))
        profiles.extend(self.find_by_dob(p.dob))
        return profiles
