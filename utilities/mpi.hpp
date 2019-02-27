// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2019  Jonas Greitemann, Ke Liu, and Lode Pollet

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <alps/utilities/mpi.hpp>

#include <stdexcept>

namespace {

    template <typename...>
    using void_t = void;

    template <class T, class = void>
    struct is_iterator : std::false_type { };

    template <class T>
    struct is_iterator<T, void_t<
        typename std::iterator_traits<T>::iterator_category
        >> : std::true_type {};

    template <class T>
    constexpr bool is_iterator_v = is_iterator<T>::value;

}

namespace mpi {

    using alps::mpi::communicator;
    using alps::mpi::broadcast;

    template <typename T>
    void send(communicator const& comm,
        T const* vals,
        size_t count,
        int dest,
        int tag)
    {
        MPI_Send(vals, count, alps::mpi::detail::mpi_type<T>(), dest, tag, comm);
    }

    template <typename ContiguousIterator,
              typename = std::enable_if_t<is_iterator_v<ContiguousIterator>>>
    void send(communicator const& comm,
        ContiguousIterator begin,
        ContiguousIterator end,
        int dest,
        int tag)
    {
        send(comm, &(*begin), end - begin, dest, tag);
    }

    template <typename T>
    void send(communicator const& comm,
        T const& val,
        int dest,
        int tag)
    {
        send(comm, &val, 1, dest, tag);
    }

    inline void send(communicator const& comm, int dest, int tag) {
        send<int>(comm, nullptr, 0, dest, tag);
    }

    template <typename T>
    int receive(communicator const& comm,
        T * vals,
        size_t count,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG)
    {
        MPI_Status status;
        MPI_Recv(vals, count, alps::mpi::detail::mpi_type<T>(), source, tag, comm, &status);
        return status.MPI_SOURCE;
    }

    template <typename ContiguousIterator,
              typename = std::enable_if_t<is_iterator_v<ContiguousIterator>>>
    int receive(communicator const& comm,
        ContiguousIterator begin,
        ContiguousIterator end,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG)
    {
        return receive(comm, &(*begin), end - begin, source, tag);
    }

    template <typename T>
    int receive(communicator const& comm,
        T & val,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG)
    {
        return receive(comm, &val, 1, source, tag);
    }

    inline int receive(communicator const& comm,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG)
    {
        return receive(comm, static_cast<int *>(nullptr), 0, source, tag);
    }

}
