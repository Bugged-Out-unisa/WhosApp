from typing import Tuple

from simple_term_menu import TerminalMenu


class PlaceholderUserManager:
    # nome predefinito dell'utente generico (hardcoded)
    DEFAULT_PLACEHOLDER = "other"

    # Opzioni per la gestione dell'utente generico
    __OPTIONS = (
        "Eliminare Utente generico",
        "Soprannominare Utente generico",
        "Procedere senza modifiche"
    )

    def __init__(self, alias_file=None):
        self.alias_file = alias_file

    def selection(self) -> Tuple[str, bool]:
        # Se non c'è un file degli alias, restituisce il nome predefinito dell'utente generico e False
        if self.alias_file is None:
            return self.DEFAULT_PLACEHOLDER, False

        # Mostra il menu delle opzioni per l'utente generico
        print("Selezione azione per l'utente generico: ")
        menu = TerminalMenu(self.__OPTIONS)
        choice = menu.show()

        # Gestione della selezione dell'utente generico
        if choice == 0:
            print("Eliminazione utente generico")
            return self.DEFAULT_PLACEHOLDER, True
        elif choice == 1:
            new_name = input("Inserire il nuovo nome per l'utente generico: ")
            return new_name, True
        else:
            # Procedi senza modificare l'utente generico e restituisci il nome predefinito
            print(f"Procedo senza modifiche. L'utente generico sarà chiamato \"{self.DEFAULT_PLACEHOLDER}\"")
            return self.DEFAULT_PLACEHOLDER, False
