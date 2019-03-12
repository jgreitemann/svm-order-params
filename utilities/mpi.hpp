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

    struct no_op_t {
        void operator()() const {}
    };

}

namespace mpi {

    using alps::mpi::communicator;
    using alps::mpi::broadcast;

    struct environment {

        enum struct threading {
            single = MPI_THREAD_SINGLE,
            funneled = MPI_THREAD_FUNNELED,
            serialized = MPI_THREAD_SERIALIZED,
            multiple = MPI_THREAD_MULTIPLE,
        };

        static void abort(int rc=0)
        {
            MPI_Abort(MPI_COMM_WORLD,rc);
        }

        static bool initialized()
        {
            int ini;
            MPI_Initialized(&ini);
            return ini;
        }

        static bool finalized()
        {
            int fin;
            MPI_Finalized(&fin);
            return fin;
        }

        environment(int& argc, char**& argv,
            threading thr = threading::single,
            bool abort_on_exception=true)
        : initialized_(false)
        , abort_on_exception_(abort_on_exception)
        {
            if (!initialized()) {
                int prov;
                MPI_Init_thread(&argc, &argv, static_cast<int>(thr), &prov);
                if (prov < static_cast<int>(thr))
                    throw std::runtime_error(
                        "requested threading mode not supported by MPI "
                        "implementation");
                initialized_=true;
            }
        }

        environment(threading thr = threading::single,
            bool abort_on_exception=true)
        : initialized_(false)
        , abort_on_exception_(abort_on_exception)
        {
            if (!initialized()) {
                int prov;
                MPI_Init_thread(nullptr, nullptr, static_cast<int>(thr), &prov);
                if (prov < static_cast<int>(thr))
                    throw std::runtime_error(
                        "requested threading mode not supported by MPI "
                        "implementation");
                initialized_=true;
            }
        }

        ~environment()
        {
            if (!initialized_) return; // we are not in control, don't mess up other's logic.
            if (finalized()) return; // MPI is finalized --- don't touch it.
            if (abort_on_exception_ && std::uncaught_exception()) {
                this->abort(255); // FIXME: make the return code configurable?
            }
            MPI_Finalize();
        }
    private:
        bool initialized_;
        bool abort_on_exception_;
    };

    inline void barrier(communicator const& comm) {
        MPI_Barrier(comm);
    }

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

    template <typename T, typename Func = no_op_t>
    void spin_send(communicator const& comm,
        T const* vals,
        size_t count,
        int dest,
        int tag,
        Func && spin_op = no_op_t{})
    {
        MPI_Request request;
        int flag;
        MPI_Isend(vals, count, alps::mpi::detail::mpi_type<T>(), dest, tag,
            comm, &request);
        while (MPI_Test(&request, &flag, MPI_STATUS_IGNORE), !flag)
            spin_op();
    }

    template <typename ContiguousIterator,
              typename Func = no_op_t,
              typename = std::enable_if_t<is_iterator_v<ContiguousIterator>>>
    void spin_send(communicator const& comm,
        ContiguousIterator begin,
        ContiguousIterator end,
        int dest,
        int tag,
        Func && spin_op = no_op_t{})
    {
        spin_send(comm, &(*begin), end - begin, dest, tag,
            std::forward<Func>(spin_op));
    }

    template <typename T, typename Func = no_op_t>
    void spin_send(communicator const& comm,
        T const& val,
        int dest,
        int tag,
        Func && spin_op = no_op_t{})
    {
        spin_send(comm, &val, 1, dest, tag, std::forward<Func>(spin_op));
    }

    template <typename Func = no_op_t>
    inline void spin_send(communicator const& comm,
        int dest,
        int tag,
        Func && spin_op = no_op_t{})
    {
        spin_send<int>(comm, nullptr, 0, dest, tag,
            std::forward<Func>(spin_op));
    }

    template <typename T, typename Func = no_op_t>
    int spin_receive(communicator const& comm,
        T * vals,
        size_t count,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG,
        Func && spin_op = no_op_t{})
    {
        MPI_Request request;
        MPI_Status status;
        int flag;
        MPI_Irecv(vals, count, alps::mpi::detail::mpi_type<T>(), source, tag,
            comm, &request);
        while (MPI_Test(&request, &flag, &status), !flag)
            spin_op();
        return status.MPI_SOURCE;
    }

    template <typename ContiguousIterator,
              typename Func = no_op_t,
              typename = std::enable_if_t<is_iterator_v<ContiguousIterator>>>
    int spin_receive(communicator const& comm,
        ContiguousIterator begin,
        ContiguousIterator end,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG,
        Func && spin_op = no_op_t{})
    {
        return spin_receive(comm, &(*begin), end - begin, source, tag,
            std::forward<Func>(spin_op));
    }

    template <typename T, typename Func = no_op_t>
    int spin_receive(communicator const& comm,
        T & val,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG,
        Func && spin_op = no_op_t{})
    {
        return spin_receive(comm, &val, 1, source, tag,
            std::forward<Func>(spin_op));
    }

    template <typename Func = no_op_t>
    inline int spin_receive(communicator const& comm,
        int source = MPI_ANY_SOURCE,
        int tag = MPI_ANY_TAG,
        Func && spin_op = no_op_t{})
    {
        return spin_receive(comm, static_cast<int *>(nullptr), 0, source, tag,
            std::forward<Func>(spin_op));
    }

    communicator split_communicator(communicator const& comm,
        int color,
        int key = 0)
    {
        MPI_Comm sub;
        MPI_Comm_split(comm, color, key, &sub);
        return {sub, alps::mpi::take_ownership};
    }

}
