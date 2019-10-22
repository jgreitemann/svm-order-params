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

#include <chrono>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <alps/utilities/mpi.hpp>


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

namespace tksvm {
namespace mpi {

    using alps::mpi::communicator;
    using alps::mpi::broadcast;
    using alps::mpi::all_reduce;

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
        MPI_Ssend(vals, count, alps::mpi::detail::mpi_type<T>(), dest, tag, comm);
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
        int source,
        int tag)
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
        int source,
        int tag)
    {
        return receive(comm, &val, 1, source, tag);
    }

    inline int receive(communicator const& comm,
        int source,
        int tag)
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
        MPI_Issend(vals, count, alps::mpi::detail::mpi_type<T>(), dest, tag,
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
    void spin_send(communicator const& comm,
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
        int source,
        int tag,
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
        int source,
        int tag,
        Func && spin_op = no_op_t{})
    {
        return spin_receive(comm, &(*begin), end - begin, source, tag,
            std::forward<Func>(spin_op));
    }

    template <typename T, typename Func = no_op_t>
    int spin_receive(communicator const& comm,
        T & val,
        int source,
        int tag,
        Func && spin_op = no_op_t{})
    {
        return spin_receive(comm, &val, 1, source, tag,
            std::forward<Func>(spin_op));
    }

    template <typename Func = no_op_t>
    int spin_receive(communicator const& comm,
        int source,
        int tag,
        Func && spin_op = no_op_t{})
    {
        return spin_receive(comm, static_cast<int *>(nullptr), 0, source, tag,
            std::forward<Func>(spin_op));
    }

    template <typename ContiguousIterator>
    void broadcast(communicator const& comm,
        ContiguousIterator begin,
        ContiguousIterator end,
        int root)
    {
        broadcast(comm, &(*begin), end - begin, root);
    }

    template <typename T>
    void all_gather(communicator const& comm,
        T const* send_vals,
        size_t send_count,
        T * recv_vals,
        size_t recv_count)
    {
        MPI_Allgather(send_vals, send_count, alps::mpi::detail::mpi_type<T>{},
            recv_vals, recv_count, alps::mpi::detail::mpi_type<T>{}, comm);
    }

    template <typename ContiguousIterator>
    auto all_gather(communicator const& comm,
        ContiguousIterator send_begin,
        ContiguousIterator send_end,
        ContiguousIterator recv_begin)
        -> ContiguousIterator
    {
        std::vector<int> recv_counts(comm.size());
        int send_count = send_end - send_begin;
        all_gather(comm, &send_count, 1, recv_counts.data(), 1);

        std::vector<int> displ(recv_counts.size(), 0);
        for (size_t i = 1; i < displ.size(); ++i)
            displ[i] = displ[i - 1] + recv_counts[i - 1];
        using T = typename ContiguousIterator::value_type;
        MPI_Allgatherv(&(*send_begin), send_count,
            alps::mpi::detail::mpi_type<T>{}, &(*recv_begin),
            recv_counts.data(), displ.data(), alps::mpi::detail::mpi_type<T>{},
            comm);
        return recv_begin + (displ.back() + recv_counts.back());
    }

    struct mutex {
        mutex(communicator const& comm, int tag = 0)
            : comm{comm}
            , lock_tag{894323 + 2 * tag}
            , unlock_tag{894323 + 2 * tag + 1}
        {
            if (comm.rank() == 0)
                thread = std::thread{[&] {
                    int proc, request;
                    while (proc = mpi::spin_receive(comm, request,
                        MPI_ANY_SOURCE, lock_tag, spinner), request)
                    {
                        mpi::spin_receive(comm, proc, unlock_tag, spinner);
                    }
                }};
        }

        mutex(mutex const&) = delete;
        mutex& operator=(mutex const&) = delete;
        mutex(mutex &&) = delete;
        mutex& operator=(mutex &&) = delete;

        ~mutex() {
            mpi::barrier(comm);
            if (comm.rank() == 0) {
                mpi::spin_send(comm, 0, 0, lock_tag, spinner);
                thread.join();
            }
        }

        void lock() {
            mpi::spin_send(comm, 1, 0, lock_tag, spinner);
        }

        void unlock() {
            mpi::spin_send(comm, 0, unlock_tag, spinner);
        }

    private:
        communicator const& comm;
        const int lock_tag;
        const int unlock_tag;

        std::thread thread;

        static void spinner() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };

    inline communicator split_communicator(communicator const& comm,
        int color,
        int key = 0)
    {
        MPI_Comm sub;
        MPI_Comm_split(comm, color, key, &sub);
        return {sub, alps::mpi::take_ownership};
    }

}
}
